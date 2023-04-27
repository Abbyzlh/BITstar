//
// Created by zhou on 23-4-26.
//

#include "../include/BatchInformedTree.h"
#include <iostream>
#include <chrono>
#include <random>
#include <functional>
#include <cmath>
#include <cfloat>

BatchInformedTree::BatchInformedTree(int dimension,int batchSize) {
    //Initialize
    this->findSolution= false;
    this->batchSize=batchSize;
    this->dimension=dimension;
    this->eta= Eta(dimension);
    this->zeta= Zeta(dimension);
    this->tree.V.push_back(start);
    this->X_samples.push_back(goal);
    this->radius=DBL_MAX;
    g_T.emplace(start,0);
    g_T.emplace(goal,DBL_MAX);
}

#pragma region RandomSample
//generate a random number in uniform distribution of [0,1]
double BatchInformedTree::GetRandomNumber(){
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    auto real_rand = std::bind(std::uniform_real_distribution<double>(0,1),std::mt19937(seed));
    return real_rand();
}

//generate a random number in normal distribution
double BatchInformedTree::GetRandomNumberGaussian(double u, double sigma) {
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    auto real_rand = std::bind(std::normal_distribution<double>(u,sigma),std::mt19937(seed));
    return real_rand();
}

Vertex BatchInformedTree::SampleUniform() {
    Vertex config;
    double pos_x, pos_y, pos_z;
    double rot_x, rot_y, rot_z;
    pos_x=x_max*GetRandomNumber()+x_min*(1-GetRandomNumber());
    pos_y=y_max*GetRandomNumber()+y_min*(1-GetRandomNumber());
    pos_z=z_max*GetRandomNumber()+z_min*(1-GetRandomNumber());
    rot_x=rx_max*GetRandomNumber()+rx_min*(1-GetRandomNumber());
    rot_y=ry_max*GetRandomNumber()+ry_min*(1-GetRandomNumber());
    rot_z=rz_max*GetRandomNumber()+rz_min*(1-GetRandomNumber());
    config.push_back(pos_x);
    config.push_back(pos_y);
    config.push_back(pos_z);
    config.push_back(rot_x);
    config.push_back(rot_y);
    config.push_back(rot_z);
    return config;
}
#pragma endregion


#pragma region r-disc RGG
int BatchInformedTree::Card(std::vector<Vertex> V) {
    return V.size();
}

double BatchInformedTree::Zeta(int dimension) {
    zeta= pow(M_PI,dimension/2) / tgamma(dimension/2+1);
    return zeta;
}

int BatchInformedTree::Eta(int dimension) {
    if(dimension==2)  return 2;  //论文参数 2D 环境选取eta为2, 4D eta为8, 6D eta为10
    else if(dimension==4) return 8;
    else return 10;
}

std::vector<Vertex> BatchInformedTree::Xf_hat(std::vector<Vertex> X) {
    std::vector<Vertex> prospectivePts;
    double c_best= g_T[goal];
    for (auto x: X) {
        if(F_hat(x)<=c_best){
            prospectivePts.push_back(x);
        }
    }
    return prospectivePts;
}

int BatchInformedTree::LebesgueMeasure(std::vector<Vertex> Xf) {
    return Xf.size();
}

double BatchInformedTree::RadiusOfRGG(int q, int lambda_X) {
    int n=this->dimension;
    radius=2*eta*pow((1+(double)(1.0/n))*(lambda_X/this->zeta)*(log(q)/q),(double)(1.0/n));
    return radius;
}

#pragma endregion


#pragma region COST FUNCS
double BatchInformedTree::dist(std::vector<double> vec1, std::vector<double> vec2) {
    if (vec1.size()!=vec2.size()){
        std::cout<<"wrong vertex input"<<std::endl;
        return -1;
    }
    double dist=0;
    for (int i = 0; i < vec1.size(); ++i) {
        dist += pow((vec1[i]-vec2[i]),2);
    }
    dist= sqrt(dist);
    return dist;
}
double BatchInformedTree::G_hat(Vertex v) {
    return dist(v,start);
}
double BatchInformedTree::H_hat(Vertex v) {
    return dist(goal,v);
}
double BatchInformedTree::F_hat(Vertex v) {
    return G_hat(v)+ H_hat(v);
}
double BatchInformedTree::C_calc(Vertex v, Vertex w) {
    /// todo
    return 0;
}
double BatchInformedTree::C_hat(Vertex v, Vertex w) {
    return dist(w,v);
}

#pragma endregion


void BatchInformedTree::BIT() {
    //line 4-9
    if (!Q_e.size()&&!Q_v.size()){
        Prune(g_T[goal]);
        for (int i = 0; i < batchSize; ++i) {
            X_samples.push_back(SampleUniform());
        }
        V_old=tree.V;
        Q_v=tree.V;
        Xf= Xf_hat(Q_v);  //X的取值？Q_v? tree.V?
        this->radius= RadiusOfRGG(Card(tree.V)+Card(X_samples), LebesgueMeasure(Xf));
    }
    //line 10-11
    while (BestQueueValue_Qv(Q_v)<= BestQueueValue_Qe(Q_e)){
        ExpandNextVertex(bestVertexInQueue);
    }
    //line 12
    Vertex vm=bestEdgeInQueue.begin, xm=bestEdgeInQueue.end;
    //line 13
    for (auto it = Q_e.begin(); it !=Q_e.end() ; ) {
        if(it->begin==vm && it->end==xm){
            it=Q_e.erase(it);
        }
        else ++it;
    }
    //line 14
    if (g_T[vm]+ C_hat(vm,xm)+ H_hat(xm)<g_T[goal]){
        double actualCost=C_calc(vm,xm);
        //line 15
        if (G_hat(vm)+actualCost + H_hat(xm)<g_T[goal]){
            //line 16
            if (g_T[vm]+actualCost <g_T[xm]){
                //line 17-18
                if(std::find(tree.V.begin(),tree.V.end(),xm)!=tree.V.end()){
                    for (auto it=tree.E.begin();it!=tree.E.end();){
                        if(it->end==xm){
                            it=tree.E.erase(it);
                        }
                        else ++it;
                    }
                }
                else{   //line 19-21
                    auto it= std::find(X_samples.begin(),X_samples.end(),xm);
                    X_samples.erase(it);
                    tree.V.push_back(xm);
                    Q_v.push_back(xm);
                }
                //line 22
                tree.E.push_back(bestEdgeInQueue);
                g_T[xm]=g_T[vm]+actualCost;
                //line 23
                for (auto it = Q_e.begin(); it !=Q_e.end() ; ) {
                    if(it->end==xm && (g_T[it->begin]+ C_hat(it->begin,it->end)>=g_T[it->end])){
                        it=Q_e.erase(it);
                    }
                    else ++it;
                }
            }
        }
    }    else{
        Q_e.resize(0);
        Q_v.resize(0);
    }
}


void BatchInformedTree::ExpandNextVertex(Vertex v) {
    X_near.resize(0);
    //line 1
    auto it= std::find(Q_v.begin(),Q_v.end(),v);
    Q_v.erase(it);
    //line 2
    for(auto x:X_samples){
        if(dist(x,v)<=radius){
            X_near.push_back(x);
        }
    }
    //line 3
    for (auto w:tree.V) {
        for (auto x: X_near) {
            if(G_hat(w) + C_hat(w, x) + H_hat(x) < g_T[goal]){
                Edge newEdge;
                newEdge.begin=w, newEdge.end=x;
                Q_e.push_back(newEdge);
            }
            g_T[x]=DBL_MAX;  //添加
        }
    }
    //line 4-6
    if(std::find(V_old.begin(),V_old.end(),v)!=V_old.end()){
        //line 5
        for (auto w:tree.V) {
            if (dist(w,v)<=radius){
                V_near.push_back(w);
            }
        }
        //line 6
        for(auto v:tree.V){
            for(auto w: V_near){
                bool belong2E=false;
                for (auto it=tree.E.begin();it!=tree.E.end();) {
                    if(it->begin==v && it->end==w)  belong2E=true;
                }
                if(belong2E) continue;
                if(G_hat(v)+ C_hat(v,w)+ H_hat(w)<g_T[goal]
                    && g_T[v]+ C_hat(v,w)<g_T[w]){
                    Edge newEdge;
                    newEdge.begin=v, newEdge.end=w;
                    Q_e.push_back(newEdge);
                }
                auto iter =g_T.find(w);     //添加
                if(iter == g_T.end()) g_T[w]=DBL_MAX;    //
            }
        }
    }
}


void BatchInformedTree::Prune(double c) {
    for(auto it=X_samples.begin();it!=X_samples.end();){
        if(F_hat(*it)>=c){
            X_samples.erase(it);
        }
        else ++it;
    }
    for(auto it=tree.V.begin();it!=tree.V.end();){
        if(F_hat(*it)>c){
            tree.V.erase(it);
        }
        else ++it;
    }
    for(auto it=tree.E.begin();it!=tree.E.end();){
        if(F_hat(it->begin)>c || F_hat(it->end)>c){
            tree.E.erase(it);
        }
        else ++it;
    }
    for(auto it=tree.V.begin();it!=tree.V.end();){
        if(g_T[*it]>=DBL_MAX){
            tree.V.erase(it);
            X_samples.push_back(*it);
        }
        else ++it;
    }
}


double BatchInformedTree::BestQueueValue_Qv(std::vector<Vertex> Q_v) {
    Vertex best;
    double bestValue=DBL_MAX;
    for (auto v: Q_v) {
        double cost=g_T[v]+ H_hat(v);
        if (cost<bestValue) {
            best=v;
            bestValue=cost;
        }
    }
    bestVertexInQueue=best;
    return bestValue;
}

double BatchInformedTree::BestQueueValue_Qe(std::vector<Edge> Q_e) {
    Edge best;
    double bestValue=DBL_MAX;
    for (auto e: Q_e) {
        double cost=g_T[e.begin]+ C_hat(e.begin,e.end)+ H_hat(e.end);
        if (cost<bestValue){
            bestValue=cost;
            best=e;
        }
    }
    bestEdgeInQueue=best;
    return bestValue;
}
