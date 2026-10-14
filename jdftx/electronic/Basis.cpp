/*-------------------------------------------------------------------
Copyright 2011 Ravishankar Sundararaman
Copyright 1996-2003 Sohrab Ismail-Beigi

This file is part of JDFTx.

JDFTx is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

JDFTx is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with JDFTx.  If not, see <http://www.gnu.org/licenses/>.
-------------------------------------------------------------------*/

#include <electronic/Everything.h>
#include <electronic/SpeciesInfo.h>
#include <atomic>
#include <cstdio>
#include <cmath>

#ifdef GPU_ENABLED
#include <core/GpuUtil.h>
#endif

namespace {
size_t nextBasisGeneration()
{ static std::atomic<size_t> counter(0); return counter.fetch_add(1,std::memory_order_relaxed)+1; }
}

Basis::Basis()
{	gInfo = 0;
	iInfo = 0;
	nbasis = 0;
	cacheGeneration=nextBasisGeneration();
}

Basis::Basis(const Basis& basis) : Basis()
{	*this = basis;
}

Basis& Basis::operator=(const Basis& basis)
{	if(this==&basis) return *this;
	if(basis.iInfo) for(const auto& species:basis.iInfo->species) species->clearProjectorCache(this);
	cacheGeneration=nextBasisGeneration();
	gInfo = basis.gInfo;
	iInfo = basis.iInfo;
	nbasis = basis.nbasis;
	// IndexArray/IndexVecArray own raw managed storage; their implicit assignment
	// would alias that storage and double-free it when copied bases are destroyed.
	iGarr.init(basis.iGarr.nData(),basis.iGarr.isOnGpu());
	index.init(basis.index.nData(),basis.index.isOnGpu());
	if(iGarr.nData()) memcpy(iGarr,basis.iGarr);
	if(index.nData()) memcpy(index,basis.index);
	head = basis.head;
	return *this;
}


void Basis::setup(const GridInfo& gInfo, const IonInfo& iInfo, double Ecut, const vector3<> k)
{	//Find the indices within Ecut:
	vector3<int> iGbox;
	for(int i=0; i<3; i++)
		iGbox[i] = 1 + int(sqrt(2*Ecut) * gInfo.R.column(i).length() / (2*M_PI)) + ceil(fabs(k[i]));
	std::vector< vector3<int> > iGvec;
	std::vector<int> indexVec;
	vector3<int> iG;
	for(iG[0]=-iGbox[0]; iG[0]<=iGbox[0]; iG[0]++)
		for(iG[1]=-iGbox[1]; iG[1]<=iGbox[1]; iG[1]++)
			for(iG[2]=-iGbox[2]; iG[2]<=iGbox[2]; iG[2]++)
				if(0.5*dot(iG+k, gInfo.GGT*(iG+k)) <= Ecut)
				{	iGvec.push_back(iG);
					indexVec.push_back(gInfo.fullGindex(iG));
				}
	setup(gInfo, iInfo, indexVec, iGvec);
	logPrintf("nbasis = %lu for k = ", nbasis); k.print(globalLog, " %6.3f ");
}

void Basis::setup(const GridInfo& gInfo, const IonInfo& iInfo, const std::vector<int>& indexVec)
{	//Compute the integer G-vectors for the specified indices:
	std::vector< vector3<int> > iGvec(indexVec.size());
	int stride1 = gInfo.S[2];
	int stride0 = gInfo.S[1] * stride1;
	for(unsigned j=0; j<indexVec.size(); j++)
	{	int index = indexVec[j];
		vector3<int> iG;
		iG[0] = index/stride0; index -= stride0*iG[0];
		iG[1] = index/stride1; index -= stride1*iG[1];
		iG[2] = index;
		for(int k=0; k<3; k++)
			if(2*iG[k] > gInfo.S[k])
				iG[k] -= gInfo.S[k];
		iGvec[j] = iG;
	}
	setup(gInfo, iInfo, indexVec, iGvec);
}



void Basis::setup(const GridInfo& gInfo, const IonInfo& iInfo,
	const std::vector<int>& indexVec, const std::vector< vector3<int> >& iGvec)
{
	// Invalidate ordinary projectors before reusing basis storage. This common
	// implementation covers both cutoff-generated and custom-index setups.
	for(const auto& species:iInfo.species) species->clearProjectorCache(this);
	cacheGeneration=nextBasisGeneration();
	this->gInfo = &gInfo;
	this->iInfo = &iInfo;
	
	nbasis = iGvec.size();
	iGarr.init(nbasis);
	index.init(nbasis);
	memcpy(iGarr.data(), &iGvec[0], sizeof(vector3<int>)*nbasis);
	memcpy(index.data(), &indexVec[0], sizeof(int)*nbasis);

	//Initialize head:
	head.clear();
	for(size_t n=0; n<nbasis; n++)
		if(iGvec[n].length_squared() < 4) //selects 27 entries (basically [-1,+1]^3)
			head.push_back(n);
}

