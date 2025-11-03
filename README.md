# WBI-Analysis-for-Salt-Chemotaxis
Python scripts for WBI analysis of C. elegans salt chemotaxis.
# 处理流程
## Extract Motion Parameters
所需要的初始文件包括视频拍摄时自动生成的文件夹中的时间戳，载物台数据等，以及中线预处理后的数据
相关notion页面：https://www.notion.so/Neural-Data-Analysis-Pipelines-1dcca7f3face8094951fe13b589c0b34?source=copy_link
代码
**202501017_WBI_sigle_extract_csv.ipynb** 单文件处理
**20251031_WBIMotionExtract.ipynb** 批量抽参数处理
### 变量
**Continuous**
✔️CTX    ✔️前向运动速度和速率    ✔️头部curvature曲率    ✔️头部摆动角速度(ang_velocity)
**Discrete**
✔️静息状态
✔️前进-后退
✔️转向（包括omega turn（以前进结尾)和卷曲状态(可能处于后退的身体高曲率状态)）
<img width="1065" height="606" alt="image" src="https://github.com/user-attachments/assets/ce66e3df-2557-4e67-a4ec-1620e195fc8a" />

## Statistical Analysis
对每个数据分别进行相关性统计分析作图
相关notion页面：https://www.notion.so/Statistical-Analysis-and-Visualization-of-WBI-data-1edca7f3face80938eebe8f44d9bc883?source=copy_link
### 聚类
### 相关性分析
代码
**20251023_WBISingleAnalysis.ipynb**（单文件）
**20251018_WBIBatchAnalysis.ipynb** （多文件）
可视化：连续变量
<img width="713" height="654" alt="binsm_CTX_neuron71" src="https://github.com/user-attachments/assets/95c55067-0574-4b23-b7a6-6bb5230d9df7" />
<img width="916" height="661" alt="LineNeuCorrWithCTXHighlightedByCoilingTurn" src="https://github.com/user-attachments/assets/ab137dfd-ad27-4d37-97e9-97efc4cc06e2" />
可视化：离散变量
<img width="685" height="759" alt="neuron72AlgforwardReverseEnd" src="https://github.com/user-attachments/assets/878f6b7a-56de-4557-b8fd-50f6e5da9dab" />
<img width="1031" height="484" alt="LineNeuCorrWithRevStartHighlightedByforward" src="https://github.com/user-attachments/assets/00dd1ba1-4f07-41d4-aea9-1819dab91619" />

## Posthoc Analysis
将上一步统计分析后的结果跨数据样本总结，可视化
代码：
**20250924_WBIPoshocAnalysis_3.ipynb**
<img width="1269" height="835" alt="image" src="https://github.com/user-attachments/assets/89b89ced-c0e7-4eaa-b77d-b5257ad166e0" />
<img width="708" height="375" alt="image" src="https://github.com/user-attachments/assets/be0ec08f-5877-4178-9546-8b76866d5eab" />

<img width="1850" height="1072" alt="image" src="https://github.com/user-attachments/assets/0f390d16-ccb8-45dd-89dc-5b72094a89cd" />
<img width="2000" height="829" alt="image" src="https://github.com/user-attachments/assets/8ff18511-1a28-444a-9a08-854dd00d42cf" />

