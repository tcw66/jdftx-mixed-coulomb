#ifndef JDFTX_ELECTRONIC_BERRYPHASE_H
#define JDFTX_ELECTRONIC_BERRYPHASE_H

#include <electronic/ColumnBundle.h>

class Everything;

//! Occupied-manifold polarization and fixed-E electric enthalpy (atomic units).
//! The field uses the physical, positive-ion convention, unlike electric-field.
class BerryPhase
{
public:
	vector3<> field;
	vector3<int> polarizationBranch; //!< total-polarization offset in primitive e R / volume quanta
	bool ready = false;
	void setup(const Everything& e);
	double evaluate(const Everything& e, std::vector<ColumnBundle>* gradient=0);
	void acceptPhase(); //!< advance the phase branch only at an accepted electronic iterate
	matrix3<> latticeGradient(const Everything& e) const; //!< symmetric strain derivative at fixed Cartesian field
	void dump(const Everything& e);

private:
	struct Link
	{	int q, next, direction, string;
		std::vector<int> source; // source G index for each destination G; -1 outside basis
	};
	std::vector<Link> links;
	std::vector<double> referencePhase;
	std::vector<double> currentPhase;
	std::vector<int> phaseDirection;
	vector3<int> mesh;
	vector3<> electronic, ionic;
	double minOverlap = 1.;
	ColumnBundle forward(const ColumnBundle& input, const Everything& e, const Link& link) const;
	ColumnBundle backward(const ColumnBundle& input, const Everything& e, const Link& link) const;
};

#endif
