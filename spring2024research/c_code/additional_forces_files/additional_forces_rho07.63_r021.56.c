#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "rebound.h"
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_dilog.h>


// after editing, use these commands:
// gcc -c -O3 -fPIC additional_forces.c -o additional_forces.o
// gcc -L. -shared additional_forces.o -o additional_forces.so -lrebound -lgsl -lm

// gcc -c -O3 -fPIC additional_forces.c -o additional_forces.o -I/opt/homebrew/include/gsl -I./rebound/rebound
// gcc -L. -shared additional_forces.o -o additional_forces.so -L./rebound/rebound -L /opt/homebrew/lib -lgsl -lm 

double sigmasq_integrand_Burkert (double x) {
    return (log(1+pow(x, 2))+2*log(1+x)-2*atan(x))/(pow(x, 2)*(1+x)*(1+pow(x, 2)));
}

void NFW_DF_Petts(struct reb_simulation* const r){
    double r_0 = 21.56;
    double rho_0 = 7.63;
    
    struct reb_particle* const particles = r->particles;
    const int N = r->N;
    //double M_GC_enc;
    double M_enc;
    double radius;
    double speed;
    double particle_rad_j;
    double x;
    double rho;
    double sigmasq_integral_term;
    double sigmasq;
    double Coulomb_log;
    double C;
    double tau_df;
    double mass;
    double absdrhodr;
    double bmax;
    double bmin;

    for (int i=0;i<N;i++){
        mass = particles[i].m;
        radius = pow(pow(particles[i].x, 2)+pow(particles[i].y, 2)+pow(particles[i].z, 2), 0.5);
        speed = pow(pow(particles[i].vx, 2)+pow(particles[i].vy, 2)+pow(particles[i].vz, 2), 0.5);
        M_enc = 4*M_PI*rho_0*pow(r_0, 3)*(log(1 + radius/r_0) - (radius/r_0) / (1 + radius/r_0));
        x = radius/r_0;
        rho = rho_0 / (x * pow(1+x, 2));
        absdrhodr = fabs(rho_0/r_0 * (3*x+1)/(pow(x, 2)*pow(1+x, 3)));
        
        if (radius > 0.000001) {
            if (radius < rho/absdrhodr) {
                bmax = radius;
            }
            else {
                bmax = rho/absdrhodr;
            }
            bmin = 0.449*mass/pow(speed, 2);
            sigmasq_integral_term = M_PI*M_PI/2 + 1.5*pow(log(1+x), 2) - 0.5*log(x/(1+x)) - 3/(1+x) - log(1+x)/(1+x) - 2*log(1+x)/x + log(1+x)/(2*x*x) - 0.5/pow(1+x, 2) - 0.5/x + 3*gsl_sf_dilog(-x);
            sigmasq = 4*M_PI*0.449*rho_0*pow(r_0, 2)*x*pow(1+x, 2) * sigmasq_integral_term;

            Coulomb_log = 0.5*log(1 + pow(bmax/bmin, 2)); // 0.5*log(1 + Lambda^2)
            C = Coulomb_log * (erf(speed/sqrt(2*sigmasq)) - sqrt(2/M_PI)*speed/sqrt(sigmasq) * exp(-pow(speed, 2)/(2*sigmasq)));

            tau_df = pow(speed, 3) / (4*M_PI*pow(0.449, 2)*rho*mass*C);
            particles[i].ax -= particles[i].vx / tau_df;
            particles[i].ay -= particles[i].vy / tau_df;
            particles[i].az -= particles[i].vz / tau_df;

            // background potential
            particles[i].ax -= 0.449 * M_enc * particles[i].x/pow(radius, 3);
            particles[i].ay -= 0.449 * M_enc * particles[i].y/pow(radius, 3);
            particles[i].az -= 0.449 * M_enc * particles[i].z/pow(radius, 3);
        }
    }
}


void NFW_DF_full(struct reb_simulation* const r){
    double r_0 = 6.0;
    double rho_0 = 1.5*18.0;
    
    struct reb_particle* const particles = r->particles;
    const int N = r->N;
    double M_GC_enc;
    double M_enc;
    double radius;
    double speed;
    double particle_rad_j;
    double x;
    double rho;
    double sigmasq_integral_term;
    double sigmasq;
    double Coulomb_log;
    double C;
    double tau_df;
    double mass;

    for (int i=0;i<N;i++){
        mass = particles[i].m;
        radius = pow(pow(particles[i].x, 2)+pow(particles[i].y, 2)+pow(particles[i].z, 2), 0.5);
        speed = pow(pow(particles[i].vx, 2)+pow(particles[i].vy, 2)+pow(particles[i].vz, 2), 0.5);
        M_enc = 4*M_PI*rho_0*pow(r_0, 3)*(log(1 + radius/r_0) - (radius/r_0) / (1 + radius/r_0));
        M_GC_enc = 0.0;
        // loop over particles and add them to enclosed GC mass if they're enclosed
        for (int j=0;j<N;j++){
            particle_rad_j = pow(pow(particles[j].x, 2)+pow(particles[j].y, 2)+pow(particles[j].z, 2), 0.5);
            if (particle_rad_j <= radius+1e-5) {
                M_GC_enc += particles[j].m;
            }
        }
        // dynamical friction: cut it off if dynamical heating of GCs would stop it
        //if (radius > 0.3) {
        if (M_enc > M_GC_enc/2.0) {            
            x = radius/r_0;
            rho = rho_0 / (x * pow(1+x, 2));
            sigmasq_integral_term = M_PI*M_PI/2 + 1.5*pow(log(1+x), 2) - 0.5*log(x/(1+x)) - 3/(1+x) - log(1+x)/(1+x) - 2*log(1+x)/x + log(1+x)/(2*x*x) - 0.5/pow(1+x, 2) - 0.5/x + 3*gsl_sf_dilog(-x);
            sigmasq = 4*M_PI*0.449*rho_0*pow(r_0, 2)*x*pow(1+x, 2) * sigmasq_integral_term;
            
            Coulomb_log = 0.5*log(1 + pow(0.5*sigmasq/(0.449*mass), 2)); // 0.5*log(1 + Lambda^2)
            C = Coulomb_log * (erf(speed/sqrt(2*sigmasq)) - sqrt(2/M_PI)*speed/sqrt(sigmasq) * exp(-pow(speed, 2)/(2*sigmasq)));
            
            tau_df = pow(speed, 3) / (4*M_PI*pow(0.449, 2)*rho*mass*C);
            particles[i].ax -= particles[i].vx / tau_df;
            particles[i].ay -= particles[i].vy / tau_df;
            particles[i].az -= particles[i].vz / tau_df;
        }
        //}

        // background potential
        particles[i].ax -= 0.449 * M_enc * particles[i].x/pow(radius, 3);
        particles[i].ay -= 0.449 * M_enc * particles[i].y/pow(radius, 3);
        particles[i].az -= 0.449 * M_enc * particles[i].z/pow(radius, 3);
    }
}

// NFW background & 1/m DF
void NFW_DF_simplest(struct reb_simulation* const r){
    double r_0 = 6.0;
    double rho_0 = 1.5*18.0;
    struct reb_particle* const particles = r->particles;
    const int N = r->N;
    double radius;
    double M_enc;
    double M_GC_enc;
    double particle_rad_j;
    for (int i=0;i<N;i++){
        radius = pow(pow(particles[i].x, 2)+pow(particles[i].y, 2)+pow(particles[i].z, 2), 0.5);
        M_enc = 4*M_PI*rho_0*pow(r_0, 3)*(log(1 + radius/r_0) - (radius/r_0) / (1 + radius/r_0));
        M_GC_enc = 0.0;
        // loop over particles and add them to enclosed GC mass if they're enclosed
        for (int j=0;j<N;j++){
            particle_rad_j = pow(pow(particles[j].x, 2)+pow(particles[j].y, 2)+pow(particles[j].z, 2), 0.5);
            if (particle_rad_j < radius) {
                M_GC_enc += particles[j].m;
            }
        }
        // dynamical friction: cut it off if dynamical heating of GCs would stop it
        if (M_enc > M_GC_enc/2.0) {
            particles[i].ax -= particles[i].vx / (25.0/particles[i].m);
            particles[i].ay -= particles[i].vy / (25.0/particles[i].m);
            particles[i].az -= particles[i].vz / (25.0/particles[i].m);
        }
        // background potential
        particles[i].ax -= 0.449 * M_enc * particles[i].x/pow(radius, 3);
        particles[i].ay -= 0.449 * M_enc * particles[i].y/pow(radius, 3);
        particles[i].az -= 0.449 * M_enc * particles[i].z/pow(radius, 3);
    }
}

// NFW background potential only
void NFW(struct reb_simulation* const r){
    double r_0 = 19.54;
    double rho_0 = 11.82;
    struct reb_particle* const particles = r->particles;
    const int N = r->N;
    double radius;
    double M_enc;
    for (int i=0;i<N;i++){
        // background potential
        radius = pow(pow(particles[i].x, 2)+pow(particles[i].y, 2)+pow(particles[i].z, 2), 0.5);
        M_enc = 4*M_PI*rho_0*pow(r_0, 3)*(log(1 + radius/r_0) - (radius/r_0) / (1 + radius/r_0));
        particles[i].ax -= 0.449 * M_enc * particles[i].x/pow(radius, 3);
        particles[i].ay -= 0.449 * M_enc * particles[i].y/pow(radius, 3);
        particles[i].az -= 0.449 * M_enc * particles[i].z/pow(radius, 3);
    }
}

void Burkert_DF_Petts(struct reb_simulation* const r){
    double r_0 = 3.36;
    double rho_0 = 318.76;
    struct reb_particle* const particles = r->particles;
    const int N = r->N;
    double radius;
    double M_enc;
    double speed;
    double particle_rad_j;
    double rho;
    double sigmasq_integral;
    double error;
    size_t _;
    gsl_function F;
    double sigmasq;
    double Coulomb_log;
    double C;
    double tau_df;
    double omegasq;
    double d2psidr2;
    double r_tidal;
    double absdrhodr;
    double bmax;
    double bmin;
    double x;

    for (int i=0;i<N;i++){
        radius = pow(pow(particles[i].x, 2)+pow(particles[i].y, 2)+pow(particles[i].z, 2), 0.5);
        x = radius/r_0;
        M_enc = M_PI * rho_0 * pow(r_0, 3) * (log(1.0 + pow(x, 2)) + 2.0*log(1.0 + x) - 2.0*atan(x));
        speed = pow(pow(particles[i].vx, 2)+pow(particles[i].vy, 2)+pow(particles[i].vz, 2), 0.5);
        rho = rho_0 / ((1+x) * (1 + pow(x, 2)));
        absdrhodr = fabs(rho_0/r_0 * (3*pow(x, 2) + 2*x + 1)/(pow(1+x, 2)*pow(1 + pow(x, 2), 2)));
        omegasq = M_enc/pow(radius, 3);
        d2psidr2 = (2*M_enc/pow(radius, 3) - 4*M_PI*rho);
        r_tidal = pow(particles[i].m/(omegasq+d2psidr2), 1.0/3.0);
        
        // dynamical friction: cut it off in the core by comparing tidal radii
        if (radius > r_tidal) {
            gsl_integration_workspace * w = gsl_integration_workspace_alloc(1000);
            F.function = &sigmasq_integrand_Burkert;
            gsl_integration_qng(&F, radius/r_0, 100, 1e-2, 1e-2, &sigmasq_integral, &error, &_);
            sigmasq = M_PI*0.449*pow(rho_0, 2)*pow(r_0, 2)/rho * sigmasq_integral;
            gsl_integration_workspace_free(w);
            if (radius < rho/absdrhodr) {
                bmax = radius;
            }
            else {
                bmax = rho/absdrhodr;
            }
            bmin = 0.449*particles[i].m/pow(speed, 2);
            
            Coulomb_log = 0.5*log(1 + pow(bmax/bmin, 2)); // 0.5*log(1 + Lambda^2)
            C = Coulomb_log * (erf(speed/sqrt(2*sigmasq)) - sqrt(2/M_PI)*speed/sqrt(sigmasq) * exp(-pow(speed, 2)/(2*sigmasq)));

            tau_df = pow(speed, 3) / (4*M_PI*pow(0.449, 2)*rho*particles[i].m*C);
            particles[i].ax -= particles[i].vx / tau_df;
            particles[i].ay -= particles[i].vy / tau_df;
            particles[i].az -= particles[i].vz / tau_df;
        }
        // background potential
        particles[i].ax -= 0.449 * M_enc * particles[i].x/pow(radius, 3);
        particles[i].ay -= 0.449 * M_enc * particles[i].y/pow(radius, 3);
        particles[i].az -= 0.449 * M_enc * particles[i].z/pow(radius, 3);
    }
}

void Burkert_DF_full(struct reb_simulation* const r){
    double r_0 = 2.0;
    double R_e = 0.9;
    double rho_0 = 1.5*166.0;
    struct reb_particle* const particles = r->particles;
    const int N = r->N;
    double radius;
    double M_enc;
    double speed;
    double M_GC_enc;
    double particle_rad_j;
    double rho;
    double sigmasq_integral;
    double error;
    size_t _;
    gsl_function F;
    double sigmasq;
    double Coulomb_log;
    double C;
    double tau_df;

    for (int i=0;i<N;i++){
        radius = pow(pow(particles[i].x, 2)+pow(particles[i].y, 2)+pow(particles[i].z, 2), 0.5);
        M_enc = M_PI * rho_0 * pow(r_0, 3) * (log(1.0 + pow(radius/r_0, 2)) + 2.0*log(1.0 + radius/r_0) - 2.0*atan(radius/r_0));
        speed = pow(pow(particles[i].vx, 2)+pow(particles[i].vy, 2)+pow(particles[i].vz, 2), 0.5);
        // dynamical friction: cut it off in the core, 0.3R_e
        if (radius > 0.3 * R_e) {
            M_GC_enc = 0.0;
            // loop over particles and add them to enclosed GC mass if they're enclosed
            for (int j=0;j<N;j++){
                particle_rad_j = pow(pow(particles[j].x, 2)+pow(particles[j].y, 2)+pow(particles[j].z, 2), 0.5);
                if (particle_rad_j <= radius+1e-5) {
                    M_GC_enc += particles[j].m;
                }
            }
            // dynamical friction: cut it off if dynamical heating of GCs would stop it - turns out this is irrelevant! total GC mass too low
            if (M_enc > M_GC_enc/2.0) {
                rho = rho_0 / ((1+radius/r_0) * (1 + pow(radius/r_0, 2)));
                
                gsl_integration_workspace * w = gsl_integration_workspace_alloc(1000);
                F.function = &sigmasq_integrand_Burkert;
                gsl_integration_qng(&F, radius/r_0, 100, 1e-2, 1e-2, &sigmasq_integral, &error, &_);
                sigmasq = M_PI*0.449*pow(rho_0, 2)*pow(r_0, 2)/rho * sigmasq_integral;
                gsl_integration_workspace_free(w);
                Coulomb_log = 0.5*log(1 + pow(2*speed*speed*radius/(0.449*particles[i].m), 2)); // 0.5*log(1 + Lambda^2)
                C = Coulomb_log * (erf(speed/sqrt(2*sigmasq)) - sqrt(2/M_PI)*speed/sqrt(sigmasq) * exp(-pow(speed, 2)/(2*sigmasq)));

                tau_df = pow(speed, 3) / (4*M_PI*pow(0.449, 2)*rho*particles[i].m*C);
                particles[i].ax -= particles[i].vx / tau_df;
                particles[i].ay -= particles[i].vy / tau_df;
                particles[i].az -= particles[i].vz / tau_df;
            }
        }
        // background potential
        particles[i].ax -= 0.449 * M_enc * particles[i].x/pow(radius, 3);
        particles[i].ay -= 0.449 * M_enc * particles[i].y/pow(radius, 3);
        particles[i].az -= 0.449 * M_enc * particles[i].z/pow(radius, 3);
    }
}

// Burkert background & 1/m DF taking into account core stalling and heating effects
void Burkert_DF_simplest(struct reb_simulation* const r){
    double r_0 = 2.0;
    double rho_0 = 1.5*166.0;
    struct reb_particle* const particles = r->particles;
    const int N = r->N;
    double radius;
    double M_enc;
    double M_GC_enc;
    double particle_rad_j;
    for (int i=0;i<N;i++){
        radius = pow(pow(particles[i].x, 2)+pow(particles[i].y, 2)+pow(particles[i].z, 2), 0.5);
        M_enc = M_PI * rho_0 * pow(r_0, 3) * (log(1.0 + pow(radius/r_0, 2)) + 2.0*log(1.0 + radius/r_0) - 2.0*atan(radius/r_0));
        // dynamical friction: cut it off in the core
        if (radius > 0.3 * r_0) {
            M_GC_enc = 0.0;
            // loop over particles and add them to enclosed GC mass if they're enclosed
            for (int j=0;j<N;j++){
                particle_rad_j = pow(pow(particles[j].x, 2)+pow(particles[j].y, 2)+pow(particles[j].z, 2), 0.5);
                if (particle_rad_j <= radius) {
                    M_GC_enc += particles[j].m;
                }
            }
            // dynamical friction: cut it off if dynamical heating of GCs would stop it
            if (M_enc > M_GC_enc/2.0) {
                particles[i].ax -= particles[i].vx / (25.0/particles[i].m);
                particles[i].ay -= particles[i].vy / (25.0/particles[i].m);
                particles[i].az -= particles[i].vz / (25.0/particles[i].m);
            }
        }
        // background potential
        particles[i].ax -= 0.449 * M_enc * particles[i].x/pow(radius, 3);
        particles[i].ay -= 0.449 * M_enc * particles[i].y/pow(radius, 3);
        particles[i].az -= 0.449 * M_enc * particles[i].z/pow(radius, 3);
    }
}

// Burkert background potential only
void Burkert(struct reb_simulation* const r){
    double r_0 = 3.36;
    double rho_0 = 318.76;
    struct reb_particle* const particles = r->particles;
    const int N = r->N;
    double radius;
    double M_enc;
    for (int i=0;i<N;i++){
        // background potential
        radius = pow(pow(particles[i].x, 2)+pow(particles[i].y, 2)+pow(particles[i].z, 2), 0.5);
        M_enc = M_PI * rho_0 * pow(r_0, 3) * (log(1.0 + pow(radius/r_0, 2)) + 2.0*log(1.0 + radius/r_0) - 2.0*atan(radius/r_0));
        particles[i].ax -= 0.449 * M_enc * particles[i].x/pow(radius, 3);
        particles[i].ay -= 0.449 * M_enc * particles[i].y/pow(radius, 3);
        particles[i].az -= 0.449 * M_enc * particles[i].z/pow(radius, 3);
    }
}
