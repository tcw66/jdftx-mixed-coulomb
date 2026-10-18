/* Completed representative-q response checkpoints. GPL v3 or later; part of JDFTx.
 * Private implementation header for DFPTPhonon.cpp (native-endian format).
 */
#ifndef JDFTX_PERTURB_DFPTCHECKPOINT_H
#define JDFTX_PERTURB_DFPTCHECKPOINT_H

#include <cerrno>
#include <sys/stat.h>

const std::vector<string>& getPseudopotentialPrefixes(); // commands/ion_species.cpp

namespace {
// Head-only checked I/O. The checksum covers the header and all numerical data.
struct DFPTCheckpointStream {
	FILE* fp;
	bool writing;
	uint64_t hash=14695981039346656037ULL;
	DFPTCheckpointStream(FILE* fp,bool writing) : fp(fp),writing(writing) {}
	void bytes(void* data,size_t size) {
		if((writing ? fwrite(data,1,size,fp) : fread(data,1,size,fp))!=size)
			die_alone("Truncated DFPT checkpoint or checkpoint I/O error.\n");
		const unsigned char* p=static_cast<unsigned char*>(data);
		for(size_t i=0;i<size;i++) { hash^=p[i]; hash*=1099511628211ULL; }
	}
	void finish() {
		uint64_t expected=hash;
		if(writing) {
			if(fwrite(&expected,sizeof(expected),1,fp)!=1)
				die_alone("Cannot write DFPT checkpoint checksum.\n");
		} else {
			if(fread(&expected,sizeof(expected),1,fp)!=1 || expected!=hash || fgetc(fp)!=EOF)
				die_alone("Corrupt DFPT checkpoint: checksum or file length mismatch.\n");
		}
	}
};

class DFPTCheckpoint {
	Everything& e;
	string directory,metadata;
	std::vector<double> density;
	string filename(int iq) const {
		char suffix[64]; snprintf(suffix,sizeof(suffix),"/q-%06d.bin",iq+1);
		return directory+suffix;
	}
	void header(DFPTCheckpointStream& stream,int iq,const std::vector<int>& columns) {
		std::ostringstream out;
		out << metadata << "\nq " << iq << " columns";
		for(int col:columns) out << ' ' << col;
		std::string expected=out.str(),actual(expected.size(),'\0');
		uint64_t size=expected.size();
		stream.bytes(&size,sizeof(size));
		if(size!=expected.size()) die_alone("Incompatible DFPT checkpoint metadata length; use a new checkpoint directory.\n");
		if(stream.writing) actual=expected;
		stream.bytes(&actual[0],size);
		if(actual!=expected) {
			size_t offset=0;
			while(offset<actual.size() && actual[offset]==expected[offset]) offset++;
			size_t start=expected.rfind('\n',offset),end=expected.find('\n',offset);
			start=(start==std::string::npos) ? 0 : start+1;
			std::string detail=expected.substr(start,end-start);
			die_alone("Incompatible DFPT checkpoint inputs, geometry, pseudopotentials or symmetry plan near '%s' (byte %zu); use a new checkpoint directory.\n",detail.c_str(),offset);
		}
		// A converged density is gauge independent. Validate it numerically so
		// CPU/GPU and different MPI decompositions can share response checkpoints.
		std::vector<double> reference=density;
		stream.bytes(reference.data(),reference.size()*sizeof(double));
		if(!stream.writing) {
			double diff=0.,norm=0.;
			for(size_t i=0;i<density.size();i++) {
				double delta=reference[i]-density[i];
				diff+=delta*delta; norm+=density[i]*density[i];
			}
			double relative=sqrt(diff/std::max(norm,1e-30));
			if(!std::isfinite(relative) || relative>1e-6)
				die_alone("DFPT checkpoint ground-state density differs (relative error %.6le > 1e-6). Use the same converged SCF state or a new checkpoint directory.\n",relative);
		}
	}
public:
	void requireComplete(const std::vector<int>& independent,const MPIUtil* parent) const {
		const MPIUtil* all=parent ? parent : mpiWorld;
		if(all->isHead()) for(int iq:independent) {
			struct stat status;
			if(stat(filename(iq).c_str(),&status) || !S_ISREG(status.st_mode))
				die_alone("Missing DFPT checkpoint for representative q %d; complete all batches before merging.\n",iq+1);
		}
		bool complete=true; all->bcast(complete);
	}
	DFPTCheckpoint(Everything& e,const MPIUtil* parent,const std::vector<int>& plan) : e(e),directory(e.pertInfo.phononCheckpoint) {
		if(directory.empty()) return;
		const MPIUtil* all=parent ? parent : mpiWorld;
		if(all->isHead()) {
			if(mkdir(directory.c_str(),0777) && errno!=EEXIST)
				die_alone("Cannot create DFPT checkpoint directory '%s': %s.\n",directory.c_str(),strerror(errno));
			struct stat status;
			if(stat(directory.c_str(),&status) || !S_ISDIR(status.st_mode))
				die_alone("DFPT checkpoint path is not a directory.\n");
		}
		bool ready=true; all->bcast(ready);
		if(mpiWorld->isHead()) {
			std::ostringstream out; out.precision(17);
			out << "JDFTx DFPT response checkpoint v1\n" << e.pertInfo.phononCheckpointInput;
			out << "\nplan";
			for(int entry:plan) out << ' ' << entry;
			out << "\nfunctional " << e.exCorr.getName() << "\ncutoff " << e.cntrl.Ecut;
			for(int i=0;i<3;i++) for(int j=0;j<3;j++) out << ' ' << e.gInfo.R(i,j);
			for(int d=0;d<3;d++) out << ' ' << e.gInfo.S[d];
			if(e.gInfoWfns) for(int d=0;d<3;d++) out << ' ' << e.gInfoWfns->S[d];
			const auto& c=e.coulombParams;
			out << "\ncoulomb " << c.geometry << ' ' << (c.geometry==CoulombParams::Slab ? c.iDir : 0)
				<< ' ' << c.embed << ' ' << c.ionMargin;
			for(int d=0;d<3;d++) out << ' ' << c.embedCenter[d] << ' ' << c.Efield[d];
			// Updating fillings can change the last few bits of nElectrons with
			// MPI reduction order. Requested physical inputs are recorded above.
			out << "\nelectrons "; out.precision(12); out << e.eInfo.nElectrons; out.precision(17);
			out << ' ' << e.eInfo.nBands
				<< ' ' << e.eInfo.smearingType << ' ' << e.eInfo.smearingWidth;
			for(const auto& k:e.eInfo.qnums) out << "\nk " << k.k[0] << ' ' << k.k[1] << ' ' << k.k[2] << ' ' << k.weight;
			for(const auto& sp:e.iInfo.species) {
				FILE* fp=nullptr;
				// Resolve the same configured pseudopotential search path used by
				// SpeciesInfo::setup (e.g. ion-species SG15/$ID_ONCV_PBE.upf).
				for(const auto& prefix:getPseudopotentialPrefixes()) {
					fp=fopen((prefix+sp->potfilename).c_str(),"rb");
					if(fp) break;
				}
				if(!fp) die_alone("Cannot fingerprint DFPT pseudopotential '%s'.\n",sp->potfilename.c_str());
				uint64_t hash=14695981039346656037ULL;
				unsigned char buffer[65536]; size_t n;
				while((n=fread(buffer,1,sizeof(buffer),fp)))
					for(size_t i=0;i<n;i++) { hash^=buffer[i]; hash*=1099511628211ULL; }
				if(ferror(fp)) die_alone("Cannot read DFPT pseudopotential fingerprint.\n");
				fclose(fp);
				out << "\nspecies " << sp->name << ' ' << sp->mass << ' ' << hash;
				for(const auto& r:sp->atpos) out << "\natom " << r[0] << ' ' << r[1] << ' ' << r[2];
			}
			metadata=out.str().c_str();
			for(const auto& field:e.eVars.n) {
				const double* data=field->data();
				density.insert(density.end(),data,data+field->nElem);
			}
		}
	}
	bool load(int iq,const std::vector<int>& columns,matrix& force,std::vector<double>& residuals,
		std::vector<complexScalarFieldArray>& potentials) {
		if(directory.empty()) return false;
		FILE* fp=nullptr; bool found=false;
		if(mpiWorld->isHead()) {
			fp=fopen(filename(iq).c_str(),"rb"); found=fp;
			if(!fp && errno!=ENOENT) die_alone("Cannot read DFPT checkpoint '%s'.\n",filename(iq).c_str());
		}
		mpiWorld->bcast(found);
		if(!found) return false;
		logPrintf("Reading DFPT checkpoint: q %d (%s).\n",iq+1,filename(iq).c_str());
		residuals.resize(columns.size());
		if(e.pertInfo.phononElectronPhonon) {
			potentials.resize(columns.size());
			for(auto& potential:potentials) {
				potential.resize(e.eVars.n.size());
				for(auto& field:potential) field=complexScalarFieldData::alloc(e.gInfo);
			}
		}
		if(fp) {
			DFPTCheckpointStream stream(fp,false); header(stream,iq,columns);
			stream.bytes(force.data(),force.nRows()*force.nCols()*sizeof(complex));
			stream.bytes(residuals.data(),residuals.size()*sizeof(double));
			for(auto& potential:potentials) for(auto& field:potential)
				stream.bytes(field->data(),field->nElem*sizeof(complex));
			stream.finish(); fclose(fp);
		}
		mpiWorld->bcastData(force); mpiWorld->bcastData(residuals);
		for(double residual:residuals) if(!std::isfinite(residual) || residual<0.)
			die("Invalid DFPT checkpoint residual.\n");
		for(auto& potential:potentials) for(auto& field:potential) field->bcastData(mpiWorld);
		logPrintf("DFPT checkpoint loaded: q %d (%zu completed responses).\n",iq+1,columns.size());
		return true;
	}
	void save(int iq,const std::vector<int>& columns,matrix& force,std::vector<double>& residuals,
		std::vector<complexScalarFieldArray>& potentials) {
		if(directory.empty() || !mpiWorld->isHead()) return;
		string target=filename(iq),temporary=target+".tmp-XXXXXX";
		std::vector<char> name(temporary.begin(),temporary.end()); name.push_back('\0');
		int fd=mkstemp(name.data());
		if(fd<0) die_alone("Cannot create DFPT checkpoint temporary file.\n");
		FILE* fp=fdopen(fd,"wb");
		if(!fp) die_alone("Cannot open DFPT checkpoint temporary stream.\n");
		DFPTCheckpointStream stream(fp,true); header(stream,iq,columns);
		stream.bytes(force.data(),force.nRows()*force.nCols()*sizeof(complex));
		stream.bytes(residuals.data(),residuals.size()*sizeof(double));
		for(auto& potential:potentials) for(auto& field:potential)
			stream.bytes(field->data(),field->nElem*sizeof(complex));
		stream.finish();
		if(fflush(fp) || fsync(fd) || fclose(fp) || rename(name.data(),target.c_str()))
			die_alone("Cannot publish DFPT checkpoint '%s'.\n",target.c_str());
		logPrintf("DFPT checkpoint saved: q %d.\n",iq+1);
	}
};
}
#endif
