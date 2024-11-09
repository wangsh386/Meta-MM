# Meta-MM：元学习框架下分子特性预测的多模态表示学习

**Meta-MM: Multimodal Representation Learning for Molecular Property Prediction under a Meta-Learning Framework**

## 1. Framework

Despite the emergence of numerous methods based on sequences, graph structures, and geometric features for molecular property prediction, most existing studies focus solely on a single modality, failing to comprehensively capture the multidimensional characteristics and complex information of molecules. To address this limitation, we propose an innovative multimodal representation learning model that integrates sequence, graph, and geometric features, combined with a meta-learning strategy to enhance prediction accuracy and generalization capabilities. 
![]
In the model architecture, multimodal information is utilized for prediction, and a double-layer optimization-based meta-learning strategy is employed to transfer meta-knowledge acquired from multiple property prediction tasks to low-data target tasks. This enables the model to quickly adapt and accurately predict molecular properties with only a few samples.

## 2. Acknowledgments

We achieved significant state-of-the-art (SOTA) results on the SIDER dataset. However, on two other major task sets, although our model outperformed the baseline methods, the improvement was marginal. This is primarily due to the complexity and large size of our model, resulting in longer training times and accuracy that did not meet our expectations, thus limiting its practical significance in real-world drug discovery.

We would like to extend special thanks to Jia Jie for his contributions to this project! Throughout this lengthy process, we supported each other and maintained unconditional trust. Although we did not achieve the desired outcomes, we both learned a great deal and remained true to our initial motivations despite numerous challenges and temptations.

## 3. Requirements

paddle-bfloat==0.1.7
paddlepaddle==2.5.1
torch==1.13.0
torch-cluster==1.6.0+pt113cu117
torch-geometric==2.2.0
torch-scatter==2.1.0+pt113cu117
torch-sparse==0.6.15+pt113cu117
torch-spline-conv==1.2.1+pt113cu117
rdkit==2023.3.1

See environment.yml for details.

## 4. Usage

# 1. Direct Execution
Download the data folder containing the datasets from: [Baidu Netdisk Link](https://pan.baidu.com/s/1JOIHfUeaxG-HyIaGS3I7AQ?pwd=jq46 )
Extraction Code: jq46

Place the data folder in the same directory as Meta-MM.py.

Run the following command:

python Meta-MM.py


# 2. Processing Your Own Dataset
If you need to process your own dataset, perform the following preprocessing steps:

python build_corpus.py --in_path {data_path} --out_path {save_path}
python build_vocab.py --corpus_path {corpus_path} --out_path {save_path}
python data_3d.py --dataset {dataset_name}
