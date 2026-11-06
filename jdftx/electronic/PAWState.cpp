/* Portable fixed-ion PAW state. GNU GPL v3 or later. */
#include <electronic/Everything.h>
#include <electronic/PAW.h>
#include <fstream>
#include <iomanip>

void IonInfo::writePAWState(const string &filename) const
{
	if (!mpiWorld->isHead())
		return;
	std::ofstream out(filename.c_str());
	if (!out)
		die_alone("Cannot write PAW state '%s'.\n", filename.c_str());
	out << std::setprecision(17) << "JDFTxPAWState 1\n";
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			out << e->gInfo.R(i, j) << " ";
	out << "\n";
	size_t count = 0;
	for (auto sp : species)
		if (sp->paw)
			++count;
	out << count << "\n";
	for (auto sp : species)
		if (sp->paw)
		{
			const PAW &p = *sp->paw;
			out << sp->name.c_str() << " " << p.fingerprint << " " << sp->atpos.size() << " "
			    << p.channels.size() << "\n";
			for (size_t a = 0; a < sp->atpos.size(); a++)
			{
				for (int k = 0; k < 3; k++)
					out << sp->atpos[a][k] << " ";
				out << "\n";
				for (const matrix *mat : {&p.rho[a], &p.potential[a]})
				{
					for (size_t i = 0; i < mat->nData(); i++)
						out << mat->data()[i].real() << " ";
					out << "\n";
				}
			}
		}
	if (!out)
		die_alone("Failed writing PAW state '%s'.\n", filename.c_str());
}
void IonInfo::readPAWState(const string &filename) const
{
	std::ifstream in(filename.c_str());
	if (!in)
		die("Fixed-H PAW calculations require '%s' from the ground-state dump (State, ElecDensity, or "
		    "Vscloc).\n",
		    filename.c_str());
	std::string magic;
	int version = 0;
	in >> magic >> version;
	if (magic != "JDFTxPAWState" || version != 1)
		die("Invalid PAW state header.\n");
	auto real = [&]()
	{
		double x = 0.;
		if (!(in >> x) || !std::isfinite(x))
			die("Invalid or truncated PAW state.\n");
		return x;
	};
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			if (fabs(real() - e->gInfo.R(i, j)) > 1e-10)
				die("PAW state lattice mismatch.\n");
	size_t count = 0, expected = 0;
	in >> count;
	for (auto sp : species)
		if (sp->paw)
			++expected;
	if (count != expected)
		die("PAW state species count mismatch.\n");
	for (auto sp : species)
		if (sp->paw)
		{
			PAW &p = *sp->paw;
			std::string name, fingerprint;
			size_t atoms = 0, channels = 0;
			in >> name >> fingerprint >> atoms >> channels;
			if (name != sp->name.c_str() || fingerprint != p.fingerprint || atoms != sp->atpos.size() ||
			    channels != p.channels.size())
				die("PAW state species, dataset, or channel mismatch.\n");
			for (size_t a = 0; a < atoms; a++)
			{
				for (int k = 0; k < 3; k++)
				{
					double saved = real(), current = sp->atpos[a][k];
					if (fabs(saved - current) > 1e-10)
						die("PAW state atomic position mismatch for %s atom %zu, lattice component %d: "
						    "saved %.17g, current %.17g.\nUse the exact SCF geometry for fixed-H PAW "
						    "restarts; "
						    "symmetry may have adjusted rounded input positions.\n",
						    sp->name.c_str(), a, k, saved, current);
				}
				for (matrix *mat : {&p.rho[a], &p.potential[a]})
				{
					for (size_t i = 0; i < mat->nData(); i++)
						mat->data()[i] = real();
					if (nrm2(*mat - dagger(*mat)) > 1e-8)
						die("Non-Hermitian PAW restart density or potential.\n");
				}
			}
		}
	std::string tail;
	if (in >> tail)
		die("Unexpected trailing PAW restart data.\n");
	logPrintf("Read PAW on-site state from '%s'.\n", filename.c_str());
}
