# 数据来源与核对记录

核对日期：2026-10-24。下表每一条都联网核对过（PubMed E-utilities、Europe PMC 全文接口、期刊页面、国家统计局网站），注册表只引用表中核对过的数字。

| 来源 | 标识 | 怎么核对的 | 本仓库用到的数字 |
|---|---|---|---|
| 国家统计局《中国人口普查年鉴-2020》表 6-4 全国分年龄、性别的死亡人口状况（2019.11.1-2020.10.31） | [A0604.xls](https://www.stats.gov.cn/sj/pcsj/rkpc/7rp/zk/html/A0604.xls) | 下载原始 xls，原样存为 `data/raw/census2020_table6-4.xls` | 0-100+ 岁逐岁平均人口、死亡人口（男女） |
| 国家统计局《中国统计年鉴2025》表 2-3 平均预期寿命 | [C02-03.jpg](https://www.stats.gov.cn/sj/ndsj/2025/html/C02-03.jpg) | 查看官方表格图片 | 2020 年 合计 77.93、男 75.37、女 80.88 |
| Sun Q, et al. Nat Commun 2025. A simulation study of the impact of population-wide lifestyle modifications on life expectancy in the Chinese population | DOI [10.1038/s41467-025-64824-x](https://doi.org/10.1038/s41467-025-64824-x)；PMID 41203624；PMC12594753 | Europe PMC 全文 XML；期刊补充材料 PDF（MOESM1）；Source Data xlsx（MOESM4） | Table 1 HR 与 95% CI；Supplementary Table 4 最终模型 β；Table 1 脚注中心化值；Supplementary Table 1 CNHS 分布；Supplementary Table 2 变量范围；Supplementary Tables 8-9 常用摄入量；Source Data Figure S6 分年龄均值/患病率；Table 2 情景结果（理想 +4.83/+5.39，现实 +2.61/+3.10，基准 30 岁 LE 45.46/47.00）；Table 3 情景定义 |
| Sun Q, et al. Lancet Public Health 2022;7(12):e994-e1004. Healthy lifestyle and life expectancy at age 30 years in the Chinese population | DOI [10.1016/S2468-2667(22)00110-4](https://doi.org/10.1016/S2468-2667(22)00110-4)；PMID 35926549；PMC7615002 | PubMed 摘要；Europe PMC 全文 | 5 项 vs 0-1 项：30 岁 LE 差 男 8.8（6.8-10.7）/ 女 8.1（6.5-9.9）；HR 0.38（0.34-0.43，男女合并）；低风险因素定义 |
| Global BMI Mortality Collaboration. Lancet 2016;388:776-786 | DOI [10.1016/S0140-6736(16)30175-1](https://doi.org/10.1016/S0140-6736(16)30175-1)；PMID 27423262 | PubMed 摘要 | BMI 15-<18.5 HR 1.51（1.43-1.59）；18.5-<20 HR 1.13（1.09-1.17）；20-25 为最低；BMI>25 每 5 单位东亚 HR 1.39（1.34-1.44）；斜率随年龄减弱（35-49 岁 1.52 vs 70-89 岁 1.21） |
| Cappuccio FP, et al. Sleep 2010;33(5):585-592 | DOI [10.1093/sleep/33.5.585](https://doi.org/10.1093/sleep/33.5.585)；PMID 20469800 | PubMed 摘要 | 短睡眠 RR 1.12（1.06-1.18）；长睡眠 RR 1.30（1.22-1.38） |
| Chow CC, Hall KD. PLoS Comput Biol 2008;4(3):e1000045 | DOI [10.1371/journal.pcbi.1000045](https://doi.org/10.1371/journal.pcbi.1000045)；PMID 18369435；PMC2266991 | Europe PMC 全文 XML | ρF = 39.5 MJ/kg；ρL = 7.6 MJ/kg；Forbes 关系 L = 10.4·ln(F/D)；示例能耗 E = 0.14L + 0.05F + 1.55 MJ/天 |
| Hall KD, et al. Lancet 2011;378:826-837 | DOI [10.1016/S0140-6736(11)60812-X](https://doi.org/10.1016/S0140-6736(11)60812-X)；PMID 21872751；PMC3880593 | PubMed 摘要；PMC 作者稿全文 | 经验法则：每天 100 kJ ≈ 最终 1 kg，约 1 年完成一半、约 3 年完成 95%；100 kg 男性每天减 2 MJ → 平台约 75 kg |
| Hall KD, et al. Cell Metab 2019;30(1):67-77 | DOI [10.1016/j.cmet.2019.05.008](https://doi.org/10.1016/j.cmet.2019.05.008)；PMID 31105044 | PubMed 摘要 | 超加工饮食期间自由进食多摄入 508 ± 106 kcal/天（只用于说明多重不良情景能量盈余的估算依据） |
| Deurenberg P, et al. Br J Nutr 1991;65(2):105-114 | DOI [10.1079/bjn19910073](https://doi.org/10.1079/bjn19910073)；PMID 2043597 | PubMed 摘要 | 成人体脂率 BF% = 1.20×BMI + 0.23×年龄 − 10.8×性别 − 5.4 |

## 使用限制

- **CKB 与 CNHS 原始数据没有使用**。两者都需要申请，CNHS 汇总数据协议限非商业研究用途（见 Nat Commun 2025 的 Data availability）。本仓库只使用论文正文、补充材料和 Source Data 里公开的汇总参数。
- **论文代码仓库 [qiufen-code/lifestyle-simulation-study](https://github.com/qiufen-code/lifestyle-simulation-study) 没有许可证**（2026-10-24 通过 GitHub API 确认 `license: null`，仓库内也没有 LICENSE 文件）。没有许可证即默认保留所有权利，所以本仓库没有复制或改写其中任何代码，只阅读了 `2-LE calculation.do` 来理解原文方法（5 年风险 = 1 − S0^exp(lp)，按 5 岁组生命表求 30 岁预期寿命），全部实现独立编写。
- Nat Commun 2025 正文采用 CC BY-NC-ND 许可；本仓库只引用其中的数字（事实数据）并注明出处，不转载正文和图表。

## 需求文档里提到、但这一版没有采用的来源

需求文档还列了 China-PAR、CHARLS、CLHLS、大庆研究、STEP、LIFE 研究、Million Veteran Program、Danish twin、Gurven–Kaplan 等。这些属于后续的疾病、功能、遗传和历史情景模块，第一版没有用到它们的数字，所以也没有逐条核对。
