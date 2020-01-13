#ifndef ElectroWeakAnalysis_RoccoR_H
#define ElectroWeakAnalysis_RoccoR_H

#include <boost/math/special_functions/erf.hpp>

struct CrystalBall{
    static const double pi;
    static const double sqrtPiOver2;
    static const double sqrt2;

    double m;
    double s;
    double a;
    double n;

    double B;
    double C;
    double D;
    double N;

    double NA;
    double Ns;
    double NC;
    double F;
    double G;
    double k;

    double cdfMa;
    double cdfPa;

    CrystalBall():m(0),s(1),a(10),n(10){
	init();
    }

    void init(){
	double fa = fabs(a);
	double ex = exp(-fa*fa/2);
	double A  = pow(n/fa, n) * ex;
	double C1 = n/fa/(n-1) * ex; 
	double D1 = 2 * sqrtPiOver2 * erf(fa/sqrt2);

	B = n/fa-fa;
	C = (D1+2*C1)/C1;   
	D = (D1+2*C1)/2;   

	N = 1.0/s/(D1+2*C1); 
	k = 1.0/(n-1);  

	NA = N*A;       
	Ns = N*s;       
	NC = Ns*C1;     
	F = 1-fa*fa/n; 
	G = s*n/fa;    

	cdfMa = cdf(m-a*s);
	cdfPa = cdf(m+a*s);
    }

    double pdf(double x) const{ 
	double d=(x-m)/s;
	if(d<-a) return NA*pow(B-d, -n);
	if(d>a) return NA*pow(B+d, -n);
	return N*exp(-d*d/2);
    }

    double pdf(double x, double ks, double dm) const{ 
	double d=(x-m-dm)/(s*ks);
	if(d<-a) return NA/ks*pow(B-d, -n);
	if(d>a) return NA/ks*pow(B+d, -n);
	return N/ks*exp(-d*d/2);
    }

    double cdf(double x) const{
	double d = (x-m)/s;
	if(d<-a) return NC / pow(F-s*d/G, n-1);
	if(d>a) return NC * (C - pow(F+s*d/G, 1-n) );
	return Ns * (D - sqrtPiOver2 * erf(-d/sqrt2));
    }

    double invcdf(double u) const{
	if(u<cdfMa) return m + G*(F - pow(NC/u, k));
	if(u>cdfPa) return m - G*(F - pow(C-u/NC, -k) );
	return m - sqrt2 * s * boost::math::erf_inv((D - u/Ns )/sqrtPiOver2);
    }
};


struct RocRes{
    enum TYPE {MC, Data, Extra};

    struct ResParams{
	double eta; 
	double kRes[2]; 
	std::vector<double> nTrk[2]; 
	std::vector<double> rsPar[3]; 
	std::vector<CrystalBall> cb;
	ResParams():eta(0){for(auto& k: kRes) k=1;}
    };

    int NETA;
    int NTRK;
    int NMIN;

    std::vector<ResParams> resol;

    RocRes();

    int etaBin(double x) const;
    int trkBin(double x, int h, TYPE T=MC) const;
    void reset();

    double rndm(int H, int F, double v) const;
    double Sigma(double pt, int H, int F) const;
    double kSpread(double gpt, double rpt, double eta, int nlayers, double w) const;
    double kSpread(double gpt, double rpt, double eta) const;
    double kSmear(double pt, double eta, TYPE type, double v, double u) const;
    double kSmear(double pt, double eta, TYPE type, double v, double u, int n) const;
    double kExtra(double pt, double eta, int nlayers, double u, double w) const;
    double kExtra(double pt, double eta, int nlayers, double u) const;
};

class RoccoR{

    private:
	enum TYPE{MC, DT};
	enum TVAR{Default, Replica, Symhes};

	static const double MPHI; 

	int NETA;
	int NPHI; 
	double DPHI;
	std::vector<double> etabin;

	struct CorParams{double M; double A;};

	struct RocOne{
	    RocRes RR;
	    std::vector<std::vector<CorParams>> CP[2];
	};

	int nset;
	std::vector<int> nmem;
	std::vector<int> tvar;
	std::vector<std::vector<RocOne>> RC;
	int etaBin(double eta) const;
	int phiBin(double phi) const;
	template <typename T> double error(T f) const;

    public:
	RoccoR(); 
	RoccoR(std::string filename); 
	void init(std::string filename);
	void reset();

	const RocRes& getRes(int s=0, int m=0) const {return RC[s][m].RR;}
	double getM(int T, int H, int F, int s=0, int m=0) const{return RC[s][m].CP[T][H][F].M;}
	double getA(int T, int H, int F, int s=0, int m=0) const{return RC[s][m].CP[T][H][F].A;}
	double getK(int T, int H, int s=0, int m=0)        const{return RC[s][m].RR.resol[H].kRes[T];}
	double kGenSmear(double pt, double eta, double v, double u, RocRes::TYPE TT=RocRes::Data, int s=0, int m=0) const;
	double kScaleMC(int Q, double pt, double eta, double phi, int s=0, int m=0) const;

	double kScaleDT(int Q, double pt, double eta, double phi, int s=0, int m=0) const;
	double kSpreadMC(int Q, double pt, double eta, double phi, double gt, int s=0, int m=0) const;
	double kSmearMC(int Q, double pt, double eta, double phi, int n, double u, int s=0, int m=0) const;

	double kScaleDTerror(int Q, double pt, double eta, double phi) const;
	double kSpreadMCerror(int Q, double pt, double eta, double phi, double gt) const;
	double kSmearMCerror(int Q, double pt, double eta, double phi, int n, double u) const;

	//old, should only be used with 2017v0
	double kScaleFromGenMC(int Q, double pt, double eta, double phi, int n, double gt, double w, int s=0, int m=0) const; 
	double kScaleAndSmearMC(int Q, double pt, double eta, double phi, int n, double u, double w, int s=0, int m=0) const;  
	double kScaleFromGenMCerror(int Q, double pt, double eta, double phi, int n, double gt, double w) const; 
	double kScaleAndSmearMCerror(int Q, double pt, double eta, double phi, int n, double u, double w) const;  
};

#endif
