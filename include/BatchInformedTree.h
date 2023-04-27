//
// Created by zhou on 23-4-26.
//

#ifndef BIT_BATCHINFORMEDTREE_H
#define BIT_BATCHINFORMEDTREE_H
#include <vector>
#include <map>
typedef std::vector<double> Vertex;
class BatchInformedTree {
public:
    BatchInformedTree();
    BatchInformedTree(int dimension,int batch_size);
    virtual ~BatchInformedTree(){};

    struct Edge{
        Vertex begin;
        Vertex end;
    };

    struct Tree{
        std::vector<Vertex> V;
        std::vector<Edge> E;
    };

    /// functional value
    bool findSolution;      // whether one solution has been found
    int dimension;          // dimension of state space
    int batchSize;          // #sample points per sampling
    int eta;                // tuning parameter
    double zeta;            // represent the Lebesgue measure of an n-dimensional unit ball.
    double radius;          // radius of RGG

    /// limitations of position
    double x_min, x_max;
    double y_min, y_max;
    double z_min, z_max;

    /// limitations of rotation in radian
    double rx_min, rx_max;
    double ry_min, ry_max;
    double rz_min, rz_max;

//    struct Vertex{
//        std::vector<double> pos;    //(x,y,z)
//        std::vector<double> cord;   //(x,y,z,theta_x,theta_y,theta_z)
//    };

    Vertex start;
    Vertex goal;

    /// Container of Vertex and Edges
    std::vector<Vertex> X_samples;
    std::vector<Vertex> X_near;
    std::vector<Vertex> Q_v;
    std::vector<Edge> Q_e;
    std::vector<Vertex> V_old;
    std::vector<Vertex> V_near;
    std::vector<Vertex> Xf;
    Tree tree;

    std::map<Vertex,double> g_T;
    Edge bestEdgeInQueue;
    Vertex bestVertexInQueue;

    /// RGG
    double GetRandomNumber();
    double GetRandomNumberGaussian(double u,double sigma);
    int Card(std::vector<Vertex> V);
    double Zeta(int dimension);
    int Eta(int dimension);
    int LebesgueMeasure(std::vector<Vertex> Xf);
    std::vector<Vertex> Xf_hat(std::vector<Vertex> X);
    double RadiusOfRGG(int q, int lambda_X);

    /// COST functions
    double dist(std::vector<double> vec1,std::vector<double> vec2);
    double F_hat(Vertex v);
    double G_hat(Vertex v);
    double H_hat(Vertex v);
    double C_hat(Vertex v,Vertex w);
    double C_calc(Vertex v,Vertex w);

    /// BestQueueValue
    double BestQueueValue_Qv(std::vector<Vertex> Q_v); //return BestQueueValueOfQv
    double BestQueueValue_Qe(std::vector<Edge> Q_e);     //return BestQueueValueOfQe

    /// Main Functions
    void BIT();
    void Prune(double c);
    void ExpandNextVertex(Vertex v);

private:
    Vertex SampleUniform();
    Vertex InformedSampleUniform();
};


#endif //BIT_BATCHINFORMEDTREE_H
