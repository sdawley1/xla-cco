/**
 * dump_reference.cpp -- golden-data harness around the group's C++ CCO code.
 *
 * Calls the UNMODIFIED cco sources (/home/sd2402/ccoptimization/cco, include
 * path only) and prints results with %.17g (decimals) or %a (hex floats).
 * Data go to the file named by --out; stdout is left to the cco library,
 * which prints progress and (for S(k)) per-k debug lines.
 *
 * Subcommands
 *   kernel : k-set, chi, energy, forces   (main.cpp:242-300 box/k-set logic)
 *   traj   : RelaxStructure_LocalGradientDescent from scratch for m = 1..P
 *   sk     : IsotropicStructureFactor(getC, n, 5, 400, Sk)
 *
 * k-window arguments (kernel, traj): either
 *   --ka KA --phi PHI [--k1a K1A]   k2 = KA / a, a = (phi/V_d(1))^(1/d)
 *                                   exactly as main.cpp:178-185, :257
 *   --k-inner X --k-outer X         raw k1, k2 (C99 hex floats via strtod)
 * Window filter (main.cpp:267-274): k1*k1 < |k|^2 < k2*k2 on Modulus2 of
 * GetKs(Config, k2, k2, 1).
 *
 * Input packs are unit-cell ConfigPacks; unless --rescale 0 they are scaled
 * by N^(1/d) as ReadConfigPack in main.cpp:100-123 / :211 does.
 *
 * Samuel J. Dawley, 11/2026
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "GeometryVector.h"
#include "PeriodicCellList.h"
#include "etc.h"
#include "StructureFactor.h"
#include "StructureOptimization.h"
#include "CollectiveCoordinatePotential.h"

typedef std::map<std::string, std::string> Args;

static Args parse_args(int argc, char ** argv)
{
	Args a;
	for (int i = 2; i + 1 < argc; i += 2) {
		std::string key = argv[i];
		if (key.compare(0, 2, "--") != 0) {
			std::fprintf(stderr, "bad argument %s\n", argv[i]);
			std::exit(2);
		}
		a[key.substr(2)] = argv[i + 1];
	}
	return a;
}

static std::string need(const Args & a, const std::string & k)
{
	auto it = a.find(k);
	if (it == a.end()) {
		std::fprintf(stderr, "missing --%s\n", k.c_str());
		std::exit(2);
	}
	return it->second;
}

static double num(const Args & a, const std::string & k, double dflt, bool required = false)
{
	auto it = a.find(k);
	if (it == a.end()) {
		if (required) need(a, k);
		return dflt;
	}
	return std::strtod(it->second.c_str(), nullptr);
}

/* Mirror of ReadConfigPack in CCOptimization/main.cpp:100-123. */
struct PackReader {
	double Rescale;
	ConfigurationPack p;
	PackReader(const std::string & prefix, double rescale) : Rescale(rescale) { p.Open(prefix); }
	Configuration operator()(size_t i)
	{
		Configuration result = p.GetConfig(i);
		result.Rescale(Rescale);
		return result;
	}
};

static void print_config(std::FILE * f, const Configuration & c, const char * tag)
{
	std::fprintf(f, "%s\n", tag);
	for (size_t p = 0; p < c.NumParticle(); p++) {
		std::fprintf(f, "S");
		for (DimensionType j = 0; j < c.GetDimension(); j++)
			std::fprintf(f, " %.17g", c.GetRelativeCoordinates(p).x[j]);
		std::fprintf(f, "\n");
	}
}

/* Everything shared by kernel and traj: reader, window, constraints. */
struct Setup {
	DimensionType dim;
	size_t N;
	double k1, k2, chi;
	std::vector<GeometryVector> ks;
	ShiftedCCPotential * pot;
	PackReader * reader;
};

static Setup make_setup(const Args & a, std::FILE * out)
{
	Setup s;
	s.dim = (DimensionType)num(a, "d", 0, true);
	s.N = (size_t)num(a, "n", 0, true);
	double rescale = (num(a, "rescale", 1) != 0.0) ? std::pow(s.N, 1.0 / (double)s.dim) : 1.0;
	s.reader = new PackReader(need(a, "pack"), rescale);
	if (a.count("ka")) {
		double phi = num(a, "phi", 0, true);
		double a_rad = std::pow(phi / (HyperSphere_Volume(s.dim, 1.0)), 1.0 / (double)s.dim);
		s.k1 = num(a, "k1a", 0.0) / a_rad;
		s.k2 = num(a, "ka", 0, true) / a_rad;
	} else {
		s.k1 = num(a, "k-inner", 0, true);
		s.k2 = num(a, "k-outer", 0, true);
	}
	Configuration Config = (*s.reader)(0);
	if (Config.NumParticle() != s.N || Config.GetDimension() != s.dim) {
		std::fprintf(stderr, "pack N/d mismatch\n");
		std::exit(3);
	}
	s.pot = new ShiftedCCPotential(s.dim);
	s.pot->ParallelNumThread = 1;
	s.ks = GetKs(Config, s.k2, s.k2, 1);
	std::fprintf(out, "KINNER %a %.17g\nKOUTER %a %.17g\n", s.k1, s.k1, s.k2, s.k2);
	for (DimensionType i = 0; i < s.dim; i++) {
		std::fprintf(out, "RECIP");
		for (DimensionType j = 0; j < s.dim; j++)
			std::fprintf(out, " %.17g", Config.GetReciprocalBasisVector(i).x[j]);
		std::fprintf(out, "\n");
	}
	double K1_modulus = s.k1 * s.k1, K2_modulus = s.k2 * s.k2, chi = 0;
	for (auto k = s.ks.begin(); k != s.ks.end(); k++) {
		double km = k->Modulus2();
		if (km > K1_modulus && km < K2_modulus) {
			std::vector<double> vals(1, 1.0);
			s.pot->AddConstraint(*k, vals);
			chi++;
			std::fprintf(out, "KVEC");
			for (DimensionType j = 0; j < s.dim; j++) std::fprintf(out, " %.17g", k->x[j]);
			std::fprintf(out, " %a\n", km);
		}
	}
	size_t M = (size_t)chi;
	chi /= s.dim * (s.N - 1);
	s.chi = chi;
	std::fprintf(out, "M %zu\nCHI %.17g\n", M, chi);
	return s;
}

static int cmd_kernel(const Args & a)
{
	std::FILE * out = std::fopen(need(a, "out").c_str(), "w");
	Setup s = make_setup(a, out);
	size_t first = (size_t)num(a, "first", 0), count = (size_t)num(a, "count", 1);
	for (size_t c = first; c < first + count; c++) {
		Configuration conf = (*s.reader)(c);
		s.pot->SetConfiguration(conf);
		double E = s.pot->Energy();
		std::vector<GeometryVector> forces;
		s.pot->AllForce(forces);
		print_config(out, conf, "CONFIG");
		std::fprintf(out, "E %.17g\n", E);
		for (size_t p = 0; p < forces.size(); p++) {
			std::fprintf(out, "F");
			for (DimensionType j = 0; j < s.dim; j++) std::fprintf(out, " %.17g", forces[p].x[j]);
			std::fprintf(out, "\n");
		}
	}
	std::fclose(out);
	return 0;
}

static int cmd_traj(const Args & a)
{
	std::FILE * out = std::fopen(need(a, "out").c_str(), "w");
	Setup s = make_setup(a, out);
	size_t first = (size_t)num(a, "first", 0), count = (size_t)num(a, "count", 1);
	size_t P = (size_t)num(a, "pmax", 256);
	for (size_t c = first; c < first + count; c++) {
		Configuration init = (*s.reader)(c);
		print_config(out, init, "CONFIG");
		for (size_t m = 1; m <= P; m++) {
			Configuration copy(init);
			RelaxStructure_LocalGradientDescent(copy, *s.pot, 0, 0, 0, 2 * m);
			s.pot->SetConfiguration(copy);
			double E = s.pot->Energy();
			std::fprintf(out, "R %zu %.17g", m, E);
			for (size_t p = 0; p < copy.NumParticle(); p++)
				for (DimensionType j = 0; j < s.dim; j++)
					std::fprintf(out, " %.17g", copy.GetRelativeCoordinates(p).x[j]);
			std::fprintf(out, "\n");
		}
	}
	std::fclose(out);
	return 0;
}

static int cmd_sk(const Args & a)
{
	std::FILE * out = std::fopen(need(a, "out").c_str(), "w");
	ConfigurationPack pack(need(a, "pack"));
	size_t count = (size_t)num(a, "count", 1);
	if ((size_t)pack.NumConfig() < count) {
		std::fprintf(stderr, "pack holds only %lld configs\n", pack.NumConfig());
		return 3;
	}
	for (size_t c = 0; c < count; c++)
		print_config(out, pack.GetConfig(c), "CONFIG");
	auto getC = [&pack](size_t i) -> Configuration { return pack.GetConfig(i); };
	std::vector<GeometryVector> Sk;
	IsotropicStructureFactor(getC, count, 5, 400, Sk);
	std::fprintf(out, "TABLE %zu\n", Sk.size());
	for (size_t r = 0; r < Sk.size(); r++)
		std::fprintf(out, "T %.17g %.17g %.17g %.17g\n", Sk[r].x[0], Sk[r].x[1], Sk[r].x[2], Sk[r].x[3]);
	std::fclose(out);
	return 0;
}

int main(int argc, char ** argv)
{
	if (argc < 2) {
		std::fprintf(stderr, "usage: dump_reference kernel|traj|sk --key value ...\n");
		return 2;
	}
	ProgramStart = std::time(nullptr);
	TimeLimit = ProgramStart + 3600L * 24 * 365; /* never trip the C++ wall-clock exit */
	Args a = parse_args(argc, argv);
	std::string cmd = argv[1];
	if (cmd == "kernel") return cmd_kernel(a);
	if (cmd == "traj") return cmd_traj(a);
	if (cmd == "sk") return cmd_sk(a);
	std::fprintf(stderr, "unknown subcommand %s\n", cmd.c_str());
	return 2;
}
