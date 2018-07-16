#include <cmath>
#include <iostream>

extern "C" double* linear_viscoelastic_model(float eta, float gamma, double *stretch, double *time, int nx){
double* q = new double[nx];
double tau = eta/gamma; // viscoelastic time constant
double dt;
double Tnc;
double Tpc;
q[0] = 0.0;
for (int ii = 0; ii < nx; ii++){
        dt = time[ii+1] - time[ii];
        Tnc = 1 - dt/(2*tau);
        Tpc = 1 + dt/(2*tau);
        q[ii+1] = pow(Tpc, -1)*(Tnc*q[ii] + gamma*(stretch[ii+1] - stretch[ii]));
    }
return q;
}


// def linear_viscoelastic_model(theta, stretch, time):
//    # unpack model parameters
//#    eta = theta['eta']
//#    gamma = theta['gamma']
//#    
//#    tau = eta/gamma # viscoelastic time constant
//#    
//#    dt = np.ones([stretch.size]) # time step
//#    dt[1:] = time[1:]-time[0:-1]
//#    n = stretch.size
//#    q = np.zeros([n,1])
//#    for kk in range(1,n):
//#        Tnc = 1 - dt[kk]/(2*tau);
//#        Tpc = 1 + dt[kk]/(2*tau);
//#        Tpcinv = Tpc**(-1);
//#        q[kk] = Tpcinv*(Tnc*q[kk-1] + gamma*(stretch[kk] - stretch[kk-1]));
//#    return q