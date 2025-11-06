# WBI-Analysis-for-Salt-Chemotaxis
Python scripts for WBI analysis of C. elegans salt chemotaxis.
# 处理流程
## Extract Motion Parameters
所需要的初始文件放在一个文件夹下
+ 钙信号数据：Calcium_traces.npy 
+ 追踪数据：视频拍摄时自动生成的文件夹中的时间戳，载物台数据等
+ 中线预处理后的数据
相关notion页面：https://www.notion.so/Neural-Data-Analysis-Pipelines-1dcca7f3face8094951fe13b589c0b34?source=copy_link <br>
代码<br>
**202501017_WBI_sigle_extract_csv.ipynb** 单文件处理<br>
**20251031_WBIMotionExtract.ipynb** 批量抽参数处理<br>
### 变量
**Continuous**<br>
✔️CTX    ✔️前向运动速度和速率    ✔️头部curvature曲率    ✔️头部摆动角速度(ang_velocity)<br>
**Discrete**<br>
✔️静息状态
✔️前进-后退
✔️转向（包括omega turn（以前进结尾)和卷曲状态(可能处于后退的身体高曲率状态)）<br>
<img width="426" height="243" alt="image" src="https://github.com/user-attachments/assets/ce66e3df-2557-4e67-a4ec-1620e195fc8a" width="200"/>

## Statistical Analysis
对每个数据分别进行相关性统计分析作图<br>
相关notion页面：https://www.notion.so/Statistical-Analysis-and-Visualization-of-WBI-data-1edca7f3face80938eebe8f44d9bc883?source=copy_link  <br>
### 聚类
### 相关性分析
代码<br>
**20251023_WBISingleAnalysis.ipynb**（单文件）<br>
**20251018_WBIBatchAnalysis.ipynb** （多文件）<br>
可视化：连续变量  
<p align="center">
  <img src="https://github.com/user-attachments/assets/95c55067-0574-4b23-b7a6-6bb5230d9df7" alt="binsm_CTX_neuron71" width="30%" style="margin-right:5px;" />
  <img src="https://github.com/user-attachments/assets/31d642ae-153e-45ee-b5e4-e43bbb548739" alt="LineNeuCorrWithCTXHighlightedByReverse" width="45%" />
</p>

可视化：离散变量  
<p align="center">
  <img src="https://github.com/user-attachments/assets/878f6b7a-56de-4557-b8fd-50f6e5da9dab" alt="neuron72AlgforwardReverseEnd" width="25%" style="margin-right:5px;" />
  <img src="https://github.com/user-attachments/assets/00dd1ba1-4f07-41d4-aea9-1819dab91619" alt="LineNeuCorrWithRevStartHighlightedByforward" width="45%" />
</p>

## Posthoc Analysis
将上一步统计分析后的结果跨数据样本总结，可视化<br>
代码：<br>
**20250924_WBIPoshocAnalysis_3.ipynb**<br>
<img alt="image" src="https://github.com/user-attachments/assets/89b89ced-c0e7-4eaa-b77d-b5257ad166e0" width="70%"/>
<img alt="image" src="https://github.com/user-attachments/assets/be0ec08f-5877-4178-9546-8b76866d5eab" width="50%" style="margin-right:5px;" />
<p align="center">
  <img src="https://github.com/user-attachments/assets/0f390d16-ccb8-45dd-89dc-5b72094a89cd" alt="image1" width="50%" style="margin-right:5px;" />
  <img src="https://github.com/user-attachments/assets/8ff18511-1a28-444a-9a08-854dd00d42cf" alt="image2" width="50%"/>
</p>

