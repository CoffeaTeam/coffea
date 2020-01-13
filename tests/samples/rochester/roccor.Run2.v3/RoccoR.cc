#ifndef ElectroWeakAnalysis_RoccoR
#define ElectroWeakAnalysis_RoccoR

#include <fstream>
#include <sstream>
#include <stdexcept>
#include "RoccoR.h"

const double CrystalBall::pi = 3.14159;
const double CrystalBall::sqrtPiOver2 = sqrt(CrystalBall::pi/2.0);
const double CrystalBall::sqrt2 = sqrt(2.0);

RocRes::RocRes(){
    reset();
}

void RocRes::reset(){
    NETA=0;
    NTRK=0;
    NMIN=0;
    std::vector<ResParams>().swap(resol);
}

int RocRes::etaBin(double eta) const{
    double abseta=fabs(eta);
    for(int i=0; i<NETA-1; ++i) if(abseta<resol[i+1].eta) return i;
    return NETA-1;
}

int RocRes::trkBin(double x, int h, TYPE T) const{
    for(int i=0; i<NTRK-1; ++i) if(x<resol[h].nTrk[T][i+1]) return i;
    return NTRK-1;
}

double RocRes::Sigma(double pt, int H, int F) const{
    double dpt=pt-45;
    const ResParams &rp = resol[H];
    return rp.rsPar[0][F] + rp.rsPar[1][F]*dpt + rp.rsPar[2][F]*dpt*dpt;
}

double RocRes::rndm(int H, int F, double w) const{
    const ResParams &rp = resol[H];
    return rp.nTrk[MC][F]+(rp.nTrk[MC][F+1]-rp.nTrk[MC][F])*w; 
}

double RocRes::kSpread(double gpt, double rpt, double eta, int n, double w) const{
    int H = etaBin(fabs(eta));
    int F = n>NMIN ? n-NMIN : 0;
    double v = rndm(H, F, w);
    int D = trkBin(v, H, Data);
    double kold = gpt / rpt;
    const ResParams &rp = resol[H];
    double u = rp.cb[F].cdf( (kold-1.0)/rp.kRes[MC]/Sigma(gpt,H,F) ); 
    double knew = 1.0 + rp.kRes[Data]*Sigma(gpt,H,D)*rp.cb[D].invcdf(u);

    if(knew<0) return 1.0;
    return kold/knew;
}


double RocRes::kSpread(double gpt, double rpt, double eta) const{
    int H = etaBin(fabs(eta));
    const auto &k = resol[H].kRes;
    double x = gpt/rpt;
    return x / (1.0 + (x-1.0)*k[Data]/k[MC]);
}

double RocRes::kSmear(double pt, double eta, TYPE type, double v, double u) const{
    int H = etaBin(fabs(eta));
    int F = trkBin(v, H); 
    const ResParams &rp = resol[H];
    double x = rp.kRes[type] * Sigma(pt, H, F) * rp.cb[F].invcdf(u);
    return 1.0/(1.0+x);
}

double RocRes::kSmear(double pt, double eta, TYPE type, double w, double u, int n) const{
    int H = etaBin(fabs(eta));
    int F = n-NMIN;
    if(type==Data) F = trkBin(rndm(H, F, w), H, Data);
    const ResParams &rp = resol[H];
    double x = rp.kRes[type] * Sigma(pt, H, F) * rp.cb[F].invcdf(u);
    return 1.0/(1.0+x);
}

double RocRes::kExtra(double pt, double eta, int n, double u, double w) const{
    int H = etaBin(fabs(eta));
    int F = n>NMIN ? n-NMIN : 0;
    const ResParams &rp = resol[H];
    double v = rp.nTrk[MC][F]+(rp.nTrk[MC][F+1]-rp.nTrk[MC][F])*w;
    int D = trkBin(v, H, Data);
    double RD = rp.kRes[Data]*Sigma(pt, H, D);
    double RM = rp.kRes[MC]*Sigma(pt, H, F);
    double x = RD>RM ? sqrt(RD*RD-RM*RM)*rp.cb[F].invcdf(u) : 0;
    if(x<=-1) return 1.0;
    return 1.0/(1.0 + x); 
}

double RocRes::kExtra(double pt, double eta, int n, double u) const{
    int H = etaBin(fabs(eta));
    int F = n>NMIN ? n-NMIN : 0;
    const ResParams &rp = resol[H];
    double d = rp.kRes[Data];
    double m = rp.kRes[MC];
    double x = d>m ? sqrt(d*d-m*m) * Sigma(pt, H, F) * rp.cb[F].invcdf(u) : 0;
    if(x<=-1) return 1.0;
    return 1.0/(1.0 + x); 
}


RoccoR::RoccoR(){}

RoccoR::RoccoR(std::string filename){
    init(filename);
}

void RoccoR::reset(){
    NETA=0;
    NPHI=0;
    std::vector<double>().swap(etabin);
    nset=0;
    std::vector<int>().swap(nmem);
    std::vector<std::vector<RocOne>>().swap(RC);

}


void RoccoR::init(std::string filename){
    std::ifstream in(filename.c_str());
    if(in.fail()) throw std::invalid_argument("RoccoR::init could not open file " + filename);

    int RMIN(0), RTRK(0), RETA(0);
    std::vector<double> BETA;

    std::string tag;
    int type, sys, mem, var, bin;	
    std::string s;
    while(std::getline(in, s)){
	std::stringstream ss(s); 
	std::string first4=s.substr(0,4);
	if(first4=="NSET"){
	    ss >> tag >> nset;
	    nmem.resize(nset);
	    tvar.resize(nset);
	    RC.resize(nset);
	}
	else if(first4=="NMEM") {
	    ss >> tag;
	    for(int i=0; i<nset; ++i) {
		ss >> nmem[i];
		RC[i].resize(nmem[i]);
	    }
	}
	else if(first4=="TVAR") {
	    ss >> tag;
	    for(int i=0; i<nset; ++i) ss >> tvar[i];
	}
	else if(first4=="RMIN") ss >> tag >> RMIN;
	else if(first4=="RTRK") ss >> tag >> RTRK;
	else if(first4=="RETA") {
	    ss >> tag >> RETA;
	    BETA.resize(RETA+1);
	    for(auto &h: BETA) ss >> h;

	}
	else if(first4=="CPHI") {
	    ss >> tag >> NPHI; 
	    DPHI=2*CrystalBall::pi/NPHI;
	}
	else if(first4=="CETA")  {
	    ss >> tag >> NETA;
	    etabin.resize(NETA+1);
	    for(auto& h: etabin) ss >> h;
	}
	else{ 
	    ss >> sys >> mem >> tag;
	    auto &rc = RC[sys][mem]; 
	    rc.RR.NETA=RETA;
	    rc.RR.NTRK=RTRK;
	    rc.RR.NMIN=RMIN;
	    auto &resol = rc.RR.resol;
	    if(resol.empty()){
		resol.resize(RETA);
		for(size_t ir=0; ir<resol.size(); ++ir){
		    auto &r = resol[ir];
		    r.eta = BETA[ir];
		    r.cb.resize(RTRK);
		    for(auto i:{0,1})r.nTrk[i].resize(RTRK+1);
		    for(auto i:{0,1,2})r.rsPar[i].resize(RTRK);
		}
	    }
	    auto &cp = rc.CP;
	    for(TYPE T:{MC,DT}){
		if(cp[T].empty()){
		    cp[T].resize(NETA);
		    for(auto &i: cp[T]) i.resize(NPHI);
		}
	    }

	    if(tag=="R"){
		ss >> var >> bin; 
		for(int i=0; i<RTRK; ++i) {
		    switch(var){
			case 0: ss >> resol[bin].rsPar[var][i]; break;
			case 1: ss >> resol[bin].rsPar[var][i]; break;
			case 2: ss >> resol[bin].rsPar[var][i]; resol[bin].rsPar[var][i]/=100; break; 
			case 3: ss >> resol[bin].cb[i].s; break; 
			case 4: ss >> resol[bin].cb[i].a; break; 
			case 5: ss >> resol[bin].cb[i].n; break; 
			default: break;
		    }
		}
	    }
	    else if(tag=="T") {
		ss >> type >> bin; 
		for(int i=0; i<RTRK+1; ++i) ss >> resol[bin].nTrk[type][i];
	    }
	    else if(tag=="F") {
		ss >> type; 
		for(int i=0; i<RETA; ++i) ss >> resol[i].kRes[type];

	    }
	    else if(tag=="C") {
		ss >> type >> var >> bin; 
		for(int i=0; i<NPHI; ++i){
		    auto &x = cp[type][bin][i];
		    if(var==0) { ss >> x.M; x.M = 1.0+x.M/100;}
		    else if(var==1){ ss >> x.A; x.A/=100; }
		}
	    }
	}
    }

    for(auto &rcs: RC)
	for(auto &rcm: rcs)
	    for(auto &r: rcm.RR.resol)
		for(auto &i: r.cb) i.init();

    in.close();
}

const double RoccoR::MPHI=-CrystalBall::pi;

int RoccoR::etaBin(double x) const{
    for(int i=0; i<NETA-1; ++i) if(x<etabin[i+1]) return i;
    return NETA-1;
}

int RoccoR::phiBin(double x) const{
    int ibin=(x-MPHI)/DPHI;
    if(ibin<0) return 0; 
    if(ibin>=NPHI) return NPHI-1;
    return ibin;
}

double RoccoR::kScaleDT(int Q, double pt, double eta, double phi, int s, int m) const{
    int H = etaBin(eta);
    int F = phiBin(phi);
    return 1.0/(RC[s][m].CP[DT][H][F].M + Q*RC[s][m].CP[DT][H][F].A*pt);
}

double RoccoR::kScaleMC(int Q, double pt, double eta, double phi, int s, int m) const{
    int H = etaBin(eta);
    int F = phiBin(phi);
    return 1.0/(RC[s][m].CP[MC][H][F].M + Q*RC[s][m].CP[MC][H][F].A*pt);
}

double RoccoR::kSpreadMC(int Q, double pt, double eta, double phi, double gt, int s, int m) const{
    const auto& rc=RC[s][m];
    int H = etaBin(eta);
    int F = phiBin(phi);
    double k=1.0/(rc.CP[MC][H][F].M + Q*rc.CP[MC][H][F].A*pt);
    return k*rc.RR.kSpread(gt, k*pt, eta);
}

double RoccoR::kSmearMC(int Q, double pt, double eta, double phi, int n, double u, int s, int m) const{
    const auto& rc=RC[s][m];
    int H = etaBin(eta);
    int F = phiBin(phi);
    double k=1.0/(rc.CP[MC][H][F].M + Q*rc.CP[MC][H][F].A*pt);
    return k*rc.RR.kExtra(k*pt, eta, n, u);
}


double RoccoR::kScaleFromGenMC(int Q, double pt, double eta, double phi, int n, double gt, double w, int s, int m) const{
    const auto& rc=RC[s][m];
    int H = etaBin(eta);
    int F = phiBin(phi);
    double k=1.0/(rc.CP[MC][H][F].M + Q*rc.CP[MC][H][F].A*pt);
    return k*rc.RR.kSpread(gt, k*pt, eta, n, w);
}

double RoccoR::kScaleAndSmearMC(int Q, double pt, double eta, double phi, int n, double u, double w, int s, int m) const{
    const auto& rc=RC[s][m];
    int H = etaBin(eta);
    int F = phiBin(phi);
    double k=1.0/(rc.CP[MC][H][F].M + Q*rc.CP[MC][H][F].A*pt);
    return k*rc.RR.kExtra(k*pt, eta, n, u, w);
}

double RoccoR::kGenSmear(double pt, double eta, double v, double u, RocRes::TYPE TT, int s, int m) const{
    return RC[s][m].RR.kSmear(pt, eta, TT, v, u);
}

template <typename T>
double RoccoR::error(T f) const{
    double sum=0;
    for(int s=0; s<nset; ++s){
	for(int i=0; i<nmem[s]; ++i) {
	    double d = f(s,i) - f(0,0); 
	    sum += d*d/nmem[s];
	}
    }
    return sqrt(sum);
}

double RoccoR::kScaleDTerror(int Q, double pt, double eta, double phi) const{
    return error([this, Q, pt, eta, phi](int s, int m) {return kScaleDT(Q, pt, eta, phi, s, m);});
}

double RoccoR::kSpreadMCerror(int Q, double pt, double eta, double phi, double gt) const{
    return error([this, Q, pt, eta, phi, gt](int s, int m){return kSpreadMC(Q, pt, eta, phi, gt, s, m);});
}

double RoccoR::kSmearMCerror(int Q, double pt, double eta, double phi, int n, double u) const{
    return error([this, Q, pt, eta, phi, n, u](int s, int m){return kSmearMC(Q, pt, eta, phi, n, u, s, m);});
}

double RoccoR::kScaleFromGenMCerror(int Q, double pt, double eta, double phi, int n, double gt, double w) const{
    return error([this, Q, pt, eta, phi, n, gt, w](int s, int m) {return kScaleFromGenMC(Q, pt, eta, phi, n, gt, w, s, m);});
}

double RoccoR::kScaleAndSmearMCerror(int Q, double pt, double eta, double phi, int n, double u, double w) const{
    return error([this, Q, pt, eta, phi, n, u, w](int s, int m) {return kScaleAndSmearMC(Q, pt, eta, phi, n, u, w, s, m);});
}

#endif

