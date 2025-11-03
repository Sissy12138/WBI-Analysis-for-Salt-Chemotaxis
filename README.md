# WBI-Analysis-for-Salt-Chemotaxis
Python scripts for WBI analysis of C. elegans salt chemotaxis.
# 处理流程
## Extract Motion Parameters
所需要的初始文件包括视频拍摄时自动生成的文件夹中的时间戳，载物台数据等，以及中线预处理后的数据
相关notion页面：https://www.notion.so/Neural-Data-Analysis-Pipelines-1dcca7f3face8094951fe13b589c0b34?source=copy_link
代码
**202501017_WBI_sigle_extract_csv.ipynb** 单文件处理
**20251031_WBIMotionExtract.ipynb** 批量抽参数处理

## Statistical Analysis
对每个数据分别进行相关性统计分析作图
相关notion页面：https://www.notion.so/Statistical-Analysis-and-Visualization-of-WBI-data-1edca7f3face80938eebe8f44d9bc883?source=copy_link
### 相关性分析
代码
**20251023_WBISingleAnalysis.ipynb**（单文件）
**20251018_WBIBatchAnalysis.ipynb** （多文件）
## Posthoc Analysis
将上一步统计分析后的结果跨数据样本总结，可视化
代码：
**20250924_WBIPoshocAnalysis_3.ipynb**
