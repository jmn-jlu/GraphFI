#include <stdio.h> 
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>
#include <cuda_runtime.h>
#include"kernel.cuh"
#include"graph.cuh"
using namespace std;
#define GPU_DEVICE 0

struct IndexedValue {
    float value;
    int index;
};
bool compareIndexedValue(const IndexedValue &a, const IndexedValue &b) {
    return a.value > b.value;
}



void CPUHITS(CsrGraph const &graph, float* hub,float* authority)
{
  const int n = graph.nodes;

  int iter_count = 0;

  float* new_hub = (float*) malloc(sizeof(float) * n);
  float* new_authority = (float*) malloc(sizeof(float) * n);

  for (int i = 0; i < n; i++){
    hub[i] = 1.0;
    authority[i]=1.0;
  }
  
  while(iter_count<10000){
    iter_count++;
    //归一化 标准化系数
    float authority_norm=0;
    float hub_norm=0;

    //存储旧值，便于比较

    for (int v = 0; v < n; v++)
    {
      //页面Authority值等于所有指向它的页面的Hub值之和
      new_authority[v] = 0.0;
      for (int j = graph.column_offsets[v]; j < graph.column_offsets[v + 1]; ++j)
      {
        int nb = graph.row_indices[j]; // v入边邻居
        new_authority[v] += hub[nb];
      }
      authority_norm+=pow(new_authority[v],2);
    
      //页面Hub值等于所有它指向的页面的Authority值之和
      new_hub[v] = 0.0;
       for (int j = graph.row_offsets[v]; j < graph.row_offsets[v + 1]; ++j)
      {
        int nb = graph.column_indices[j]; // v出边邻居
        new_hub[v] += authority[nb];
      }
      hub_norm+=pow(new_hub[v],2);
      
    }

    //归一化处理
    authority_norm=sqrt(authority_norm);
    hub_norm=sqrt(hub_norm);  
    for(int i=0;i<n;i++){
        new_authority[i]/=authority_norm;
        new_hub[i]/=hub_norm;
    }

     //比较值是否发生change
     bool change=false;
     for(int i=0;i<n;i++){ 
        if(fabs(new_authority[i]-authority[i])>0.01){
            change=true;
        }
        if(fabs(new_hub[i]-hub[i])>0.01){
            change=true;
        }
        authority[i]=new_authority[i];
        hub[i]=new_hub[i];  
     }
    if(!change)
      break;
    }
  
  //std::cout << "CPU iterations: " << iter_count << std::endl;
  
}
float l2norm(float* v1, float* v2, int n)
{
 float result = 0.0;
  for (unsigned int i = 0; i < n; ++i)
    result += (v1[i] - v2[i]) * (v1[i] - v2[i]);

  return sqrt(result);
}
float l2norm(float* v, int n)
{
  float result = 0.0;
  for (unsigned int i = 0; i < n; ++i)
    result += v[i] * v[i];

  return sqrt(result);
}

int main(int argc, char **argv)
{
 char graph_file[]= "FPC.mtx"; //图数据文件
 char outFileName[]="HITS.out"; //输出文件

  //初始化GPU
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	//printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );

    // 从MARKET格式的文件加载图数据并转化为CSR，是有向图
    CsrGraph csr_graph;
    if (BuildMarketGraph(graph_file, csr_graph,false) != 0)
      return 1;


    //打印CSR图数据
    //printCSR(csr_graph);
    
    int run_CPU =0; //用于控制是否运行CPU端的PageRank算法。
    float* reference_hub;
    float* reference_authority;
    if (run_CPU)
    {
        reference_hub = (float*) malloc(sizeof(float) * csr_graph.nodes);
        reference_authority = (float*) malloc(sizeof(float) * csr_graph.nodes);

        CPUHITS(csr_graph, reference_hub,reference_authority);
        /*
        printf("CPU Result:\n");
        for(int i=0;i<csr_graph.nodes;i++){
            printf("%f %f\n",reference_authority[i],reference_hub[i]);
        }
        printf("\n");
        */
    }
    
    float* authority=(float*) malloc(sizeof(float) * csr_graph.nodes);
    float* hub=(float*) malloc(sizeof(float) * csr_graph.nodes);
    //GPU执行
    GPUHITS(authority,hub,csr_graph.row_offsets,csr_graph.column_indices,csr_graph.column_offsets,csr_graph.row_indices,csr_graph.nodes,csr_graph.edges);
    
    //比较CPU、GPU实现结果是否一致
    if(run_CPU){
        double tol=0.1;//容忍误差
        printf("\nCorrectness testing ...\n");
        float l2error = l2norm(reference_authority, authority, csr_graph.nodes) / l2norm(reference_authority, csr_graph.nodes); 
        bool pass = l2error < tol;
        printf("authority %s! l2 error = %f\n", pass?"passed!":"failed!", l2error);
        
        l2error = l2norm(reference_hub, hub, csr_graph.nodes) / l2norm(reference_hub, csr_graph.nodes);
        pass = l2error < tol;
        printf("hub %s! l2 error = %f\n", pass?"passed!":"failed!", l2error);
        free(reference_hub);
         free(reference_authority);
        
    }

    //计算最终加权值 =0.7*authority+0.3*hub
    float* res=(float*) malloc(sizeof(float) * csr_graph.nodes);
    for(int i=0;i<csr_graph.nodes;i++){
      res[i]=0.7*authority[i]+0.3*hub[i];
    }
    
    //输出结果到输出文件
    FILE* f = fopen(outFileName, "w");
    
    // 创建一个包含res值和索引的结构体的向量
    std::vector<IndexedValue> indexedArr;
    for (int i = 0; i < csr_graph.nodes; i++) {
        indexedArr.push_back({res[i], i+1});
    }
    // 对值进行降序排序
    std::sort(indexedArr.begin(), indexedArr.end(), compareIndexedValue);
    
    for (const IndexedValue &iv : indexedArr)
    {
      fprintf(f,"%d\n",iv.index);
    }
    fclose(f);
  

  }
