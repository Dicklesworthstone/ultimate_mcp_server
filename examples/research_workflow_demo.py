#!/usr/bin/env python
"""
Advanced Research Assistant Workflow Demo

This script demonstrates a realistic research workflow using the DAG-based
workflow execution system. It processes research documents through multiple
analysis stages and produces visualizations of the results.
"""

import asyncio
import os
import sys
from collections import namedtuple  # Import namedtuple

# Add the project root to path so we can import ultimate_mcp_server
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.tools.optimization import execute_optimized_workflow
from ultimate_mcp_server.utils import get_logger  # Import get_logger
from ultimate_mcp_server.utils.display import CostTracker  # Import CostTracker

# Initialize rich console
console = Console()

# Initialize logger here so it's available in main()
logger = get_logger("example.research_workflow")

# Create a simple structure for cost tracking from dict (tokens might be missing)
TrackableResult = namedtuple(
    "TrackableResult",
    ["cost", "input_tokens", "output_tokens", "provider", "model", "processing_time"],
)

# Sample research documents
SAMPLE_DOCS = [
    """
    # The Impact of Climate Change on Coastal Communities: A Multi-Regional Analysis
    
    ## Abstract
    This comprehensive study examines the cascading effects of climate change on 570+ coastal cities globally, with projections extending to 2050. Using data from the IPCC AR6 report and economic models from the World Bank (2021), we identify adaptation costs exceeding $71 billion annually. The research incorporates satellite data from NASA's GRACE mission and economic vulnerability indices developed by Stern et al. (2019) to assess regional disparities.
    
    ## Vulnerable Regions and Economic Impact Assessment
    
    ### 1. Southeast Asia
    The Mekong Delta region, home to 17 million people, faces submersion threats to 38% of its landmass by 2050. Ho Chi Minh City has invested $1.42 billion in flood prevention infrastructure, while Bangkok's $2.3 billion flood management system remains partially implemented. The Asian Development Bank (ADB) estimates adaptation costs will reach $5.7 billion annually for Vietnam alone.
    
    ### 2. Pacific Islands
    Kiribati, Tuvalu, and the Marshall Islands face existential threats, with projected displacement of 25-35% of their populations by 2050 according to UN estimates. Australia's "Pacific Resilience Fund" ($2.1 billion) supports adaptation, but President Maamau of Kiribati has criticized its scope as "drastically insufficient." The 2022 Wellington Accords established migration pathways for climate refugees, though implementation remains fragmented.
    
    ### 3. North American Coastal Zones
    Miami-Dade County's $6 billion "Rising Above" initiative represents the largest municipal climate adaptation budget in North America. The U.S. Army Corps of Engineers projects that without intervention, coastal erosion will affect 31% of Florida's beaches by 2040. Economic models by Greenstone and Carleton (2020) indicate property devaluation between $15-27 billion in Florida alone.
    
    ## Adaptation Strategies and Cost-Benefit Analysis
    
    ### Infrastructure Hardening
    The Netherlands' Room for the River program ($2.6 billion) has demonstrated 300% ROI through prevented flood damage. Conversely, New Orleans' post-Katrina $14.5 billion levee system upgrades show more modest returns (130% ROI) due to maintenance requirements and subsidence issues highlighted by Professor Sarah Jenkins (MIT).
    
    ### Managed Retreat
    Indonesia's capital relocation from Jakarta to Borneo (est. cost $34 billion) represents the largest planned managed retreat globally. Smaller programs in Alaska (Newtok and Shishmaref villages) provide case studies with per-capita costs exceeding $380,000. Dr. Robert Chen's longitudinal studies show significant social cohesion challenges, with 47% of relocated communities reporting decreased quality-of-life metrics despite improved safety.
    
    ### Ecosystem-Based Approaches
    Vietnam's mangrove restoration initiative ($220 million) reduces storm surge impacts by 20-50% and provides $8-$20 million in annual aquaculture benefits. The Nature Conservancy's coral reef insurance programs in Mexico demonstrate innovative financing mechanisms while providing co-benefits for local tourism economies valued at $320 million annually.
    
    ## Cross-Disciplinary Implications
    
    Climate migration pathways identified by the UNHCR will increase urban population pressures in receiving cities, particularly in Manila, Dhaka, and Lagos. Healthcare systems in coastal regions report increasing cases of waterborne diseases (62% increase since 2010) and mental health challenges associated with displacement anxiety as documented by the WHO Southeast Asia regional office.
    
    ## References
    
    1. IPCC (2021). AR6 Climate Change 2021: Impacts, Adaptation and Vulnerability
    2. Stern, N., et al. (2019). Economic vulnerability indices for coastal communities
    3. Asian Development Bank. (2022). Southeast Asia Climate Adaptation Report
    4. Greenstone, M., & Carleton, T. (2020). Coastal property value projections 2020-2050
    5. Jenkins, S. (2022). Engineering limitations in climate adaptation infrastructure
    6. Chen, R. (2021). Social dimensions of community relocation programs
    7. World Health Organization. (2021). Climate change health vulnerability assessments
    """,
    """
    # Renewable Energy Transition: Economic Implications and Policy Frameworks
    
    ## Executive Summary
    This multi-phase analysis examines the economic transformation accompanying the global renewable energy transition, projecting the creation of 42.3 million new jobs by 2050 while identifying significant regional disparities and transition barriers. Drawing on data from 157 countries, this research provides comprehensive policy recommendations for equitable implementation paths.
    
    ## Methodological Framework
    
    Our modeling utilizes a modified integrated assessment model combining economic inputs from the International Energy Agency (IEA), IRENA's Renewable Jobs Tracker database, and McKinsey's Global Energy Perspective 2022. Labor market projections incorporate automation factors derived from Oxford Economics' Workforce Displacement Index, providing more nuanced job creation estimates than previous studies by Zhang et al. (2019).
    
    ## Employment Transformation Analysis by Sector
    
    ### Solar Photovoltaic Industry
    Employment projections indicate 18.7 million jobs by 2045, concentrated in manufacturing (32%), installation (41%), and operations/maintenance (27%). Regional distribution analysis reveals concerning inequities, with China capturing 41% of manufacturing roles while sub-Saharan Africa secures only 2.3% despite having 16% of global solar potential. The Skill Transferability Index suggests 73% of displaced fossil fuel workers could transition to solar with targeted 6-month reskilling programs.
    
    ### Wind Energy Sector
    Offshore wind development led by Ørsted, Vestas, and General Electric is projected to grow at 24% CAGR through 2035, creating 6.8 million jobs. Supply chain bottlenecks in rare earth elements (particularly neodymium and dysprosium) represent critical vulnerabilities, with 83% of processing controlled by three Chinese companies. Professor Tanaka's analysis suggests price volatilities of 120-350% are possible under geopolitical tensions.
    
    ### Energy Storage Revolution
    Recent lithium-ferro-phosphate (LFP) battery innovations by CATL have reduced implementation costs by 27% while boosting cycle life by 4,000 cycles. Grid-scale storage installations are projected to grow from 17GW (2022) to 220GW by 2035, employing 5.3 million in manufacturing and installation. The MIT Battery Initiative under Dr. Viswanathan has demonstrated promising alternative chemistries using earth-abundant materials that could further accelerate adoption if commercialized by 2025.
    
    ### Hydrogen Economy Emergence
    Green hydrogen production costs have declined from $5.70/kg in 2018 to $3.80/kg in 2023, with projected cost parity with natural gas achievable by 2028 according to BloombergNEF. The European Hydrogen Backbone initiative, requiring €43 billion in infrastructure investment, could generate 3.8 million jobs while reducing EU natural gas imports by 30%. Significant technological challenges remain in storage density and transport infrastructure, as highlighted in critical analyses by Professors Wilson and Leibreich.
    
    ## Transition Barriers and Regional Disparities
    
    ### Financial Constraints
    Developing economies face investment gaps of $730 billion annually according to the Climate Policy Initiative's 2022 report. The African Development Bank estimates that 72% of sub-Saharan African energy projects stall at the planning phase due to financing constraints despite IRRs exceeding 11.5%. Innovative financing mechanisms through the Global Climate Fund have mobilized only 23% of pledged capital as of Q1 2023.
    
    ### Policy Framework Effectiveness
    
    Cross-jurisdictional analysis of 87 renewable portfolio standards reveals three dominant policy approaches:
    
    1. **Carbon Pricing Mechanisms**: The EU ETS carbon price of €85/ton has driven 16.5% emissions reduction in the power sector, while Canada's escalating carbon price schedule ($170/ton by 2030) provides investment certainty. Econometric modeling by Dr. Elizabeth Warren (LSE) indicates prices must reach €120/ton to fully internalize climate externalities.
    
    2. **Direct Subsidies**: Germany's Energiewende subsidies (€238 billion cumulative) achieved 44% renewable penetration but at high consumer costs. Targeted manufacturing incentives under the U.S. Inflation Reduction Act demonstrate improved cost-efficiency with 3.2x private capital mobilization according to analysis by Resources for the Future.
    
    3. **Phased Transition Approaches**: Denmark's offshore wind cluster development model produced the highest success metrics in our analysis, reducing LCOE by 67% while creating domestic supply chains capturing 82% of economic value. This approach has been partially replicated in Taiwan and Vietnam with similar success indicators.
    
    ## Visualized Outcomes Under Various Scenarios
    
    Under an accelerated transition (consistent with 1.5°C warming), global GDP would increase by 2.4% beyond baseline by 2050, while air pollution-related healthcare costs would decline by $780 billion annually. Conversely, our "delayed action" scenario projects stranded fossil assets exceeding $14 trillion, concentrated in 8 petrostate economies, potentially triggering financial contagion comparable to 2008.
    
    ## References
    
    1. International Energy Agency. (2022). World Energy Outlook 2022
    2. IRENA. (2023). Renewable Energy Jobs Annual Review
    3. McKinsey & Company. (2022). Global Energy Perspective
    4. Zhang, F., et al. (2019). Employment impacts of renewable expansion
    5. Oxford Economics. (2021). Workforce Displacement Index
    6. Tanaka, K. (2022). Critical material supply chains in energy transition
    7. Viswanathan, V. (2023). Next-generation grid-scale storage technologies
    8. BloombergNEF. (2023). Hydrogen Economy Outlook
    9. Climate Policy Initiative. (2022). Global Landscape of Climate Finance
    10. Warren, E. (2022). Carbon pricing efficiency and distributional impacts
    11. Resources for the Future. (2023). IRA Impact Assessment
    """,
    """
    # Artificial Intelligence Applications in Healthcare Diagnostics: Implementation Challenges and Economic Analysis
    
    ## Abstract
    This comprehensive evaluation examines the integration of artificial intelligence into clinical diagnostic workflows, with particular focus on deep learning systems demonstrating 94.2% accuracy in early-stage cancer detection across 14 cancer types. The analysis spans technical validation, implementation barriers, regulatory frameworks, and economic implications based on data from 137 healthcare systems across 42 countries.
    
    ## Technological Capabilities Assessment
    
    ### Diagnostic Performance Metrics
    
    Google Health's melanoma detection algorithm demonstrated sensitivity of 95.3% and specificity of 92.7% in prospective trials, exceeding dermatologist accuracy by 18 percentage points with consistent performance across Fitzpatrick skin types I-VI. This represents significant improvement over earlier algorithms criticized for performance disparities across demographic groups as documented by Dr. Abigail Johnson in JAMA Dermatology (2021).
    
    The Mayo Clinic's AI-enhanced colonoscopy system increased adenoma detection rates from 30% to 47% in their 2022 clinical implementation study (n=3,812). This translates to approximately 68 prevented colorectal cancer cases per 1,000 screened patients according to the predictive model developed by Dr. Singh at Memorial Sloan Kettering.
    
    Stanford Medicine's deep learning algorithm for chest radiograph interpretation identified 14 pathological conditions with average AUC of 0.91, reducing false negative rates for subtle pneumothorax by 43% and pulmonary nodules by 29% in their multi-center validation study across five hospital systems with diverse patient populations.
    
    ### Architectural Innovations
    
    Recent advancements in foundation models have transformed medical AI capabilities:
    
    1. **Multi-modal integration**: Microsoft/Nuance's DAX system combines speech recognition, natural language processing, and computer vision, enabling real-time clinical documentation with 96.4% accuracy while reducing physician documentation time by 78 minutes daily according to their 16-site implementation study published in Health Affairs.
    
    2. **Explainable AI approaches**: PathAI's interpretable convolutional neural networks provide visualization of decision-making factors in histopathology, addressing the "black box" concern highlighted by regulatory agencies. Their GradCAM implementation allows pathologists to review the specific cellular features informing algorithmic conclusions, increasing adoption willingness by 67% in surveyed practitioners (n=245).
    
    3. **Federated learning**: The MELLODDY consortium's federated approach enables algorithm training across 10 pharmaceutical companies' proprietary datasets without data sharing, demonstrating how privacy-preserving computation can accelerate biomarker discovery. This approach increased available training data by 720% while maintaining data sovereignty.
    
    ## Implementation Challenges
    
    ### Clinical Workflow Integration
    
    Field studies at Massachusetts General Hospital identified five critical integration failure points that reduce AI effectiveness by 30-70% compared to validation performance:
    
    1. Alert fatigue – 52% of clinical recommendations were dismissed when AI systems generated more than 8 alerts per hour
    2. Workflow disruption – Systems requiring more than 15 seconds of additional process time saw 68% lower adoption
    3. Interface design issues – Poorly designed UI elements reduced effective utilization by 47%
    4. Confirmation bias – Clinicians were 3.4× more likely to accept AI suggestions matching their preliminary conclusion
    5. Trust calibration – 64% of clinicians struggled to appropriately weight algorithmic recommendations against their clinical judgment
    
    The Cleveland Clinic's "AI Integration Framework" addresses these challenges through graduated autonomy, contextual presentation, and embedded calibration metrics, increasing sustained adoption rates to 84% compared to the industry average of 31%.
    
    ### Data Infrastructure Requirements
    
    Analysis of implementation failures reveals data architecture as the primary barrier in 68% of stalled healthcare AI initiatives. Specific challenges include:
    
    - Legacy system integration – 73% of U.S. hospitals utilize EHR systems with insufficient API capabilities for real-time AI integration
    - Data standardization – Only 12% of clinical data meets FHIR standards without requiring significant transformation
    - Computational infrastructure – 57% of healthcare systems lack edge computing capabilities necessary for low-latency applications
    
    Kaiser Permanente's successful enterprise-wide implementation demonstrates a viable pathway through their "data fabric" architecture connecting 39 hospitals while maintaining HIPAA compliance. Their staged implementation required $43 million in infrastructure investment but delivered $126 million in annual efficiency gains by year three.
    
    ### Training Requirements for Medical Personnel
    
    Harvard Medical School's "Technology Integration in Medicine" study identified critical competency gaps among practitioners:
    
    - Only 17% of physicians could correctly interpret AI-generated confidence intervals
    - 73% overestimated algorithm capabilities in transfer scenarios
    - 81% lacked understanding of common algorithmic biases
    
    The American Medical Association's AI curriculum module has demonstrated 82% improvement in AI literacy metrics but has reached only a fraction of practitioners. Training economics present a significant barrier, with health systems reporting that comprehensive AI education requires 18-24 hours per clinician at an average opportunity cost of $5,800.
    
    ## Economic and Policy Dimensions
    
    ### Cost-Benefit Model
    
    Our economic modeling based on Medicare claims data projects net healthcare savings of $36.7 billion annually when AI diagnostic systems reach 65% market penetration. These savings derive from:
    
    - Earlier cancer detection: $14.3 billion through stage migration
    - Reduced diagnostic errors: $9.8 billion in avoided misdiagnosis costs
    - Workflow efficiency: $6.2 billion in provider time optimization
    - Avoided unnecessary procedures: $6.4 billion by reducing false positives
    
    Implementation costs average $175,000-$390,000 per facility with 3.1-year average payback periods. Rural and critical access hospitals face disproportionately longer ROI timelines (5.7 years), exacerbating healthcare disparities.
    
    ### Regulatory Framework Analysis
    
    Comparative analysis of regulatory approaches across jurisdictions reveals critical inconsistencies:
    
    | Jurisdiction | Approval Pathway | Post-Market Requirements | Algorithm Update Handling |
    |--------------|------------------|--------------------------|---------------------------|
    | FDA (US) | 510(k)/De Novo | Limited continuous monitoring | Predetermined change protocol |
    | EMA (EU) | MDR risk-based | PMCF with periodic reporting | Significant modification framework |
    | PMDA (Japan) | SAKIGAKE pathway | Mandatory registry participation | Version control system |
    | NMPA (China) | Special approval | Real-world data collection | Annual recertification |
    
    The European Medical Device Regulation's requirement for "human oversight of automated systems" creates implementation ambiguities interpreted differently across member states. The FDA's proposed "Predetermined Change Control Plan" offers the most promising framework for AI's iterative improvement nature but remains in draft status.
    
    ## Conclusions and Future Directions
    
    AI diagnosis systems demonstrate significant technical capabilities but face complex implementation barriers that transcend technological challenges. Our analysis suggests a "sociotechnical systems approach" is essential, recognizing that successful implementation depends equally on technical performance, clinical workflow integration, organizational change management, and policy frameworks.
    
    The Cleveland Clinic-Mayo Clinic consortium's phased implementation approach, beginning with augmentative rather than autonomous functionality, provides a template for successful adoption. Their experience indicates that progressive automation on a 3-5 year timeline produces superior outcomes compared to transformative implementation approaches.
    
    ## References
    
    1. Johnson, A. (2021). Demographic performance disparities in dermatological AI. JAMA Dermatology, 157(2)
    2. Mayo Clinic. (2022). AI-enhanced colonoscopy outcomes study. Journal of Gastrointestinal Endoscopy, 95(3)
    3. Singh, K. (2021). Predictive modeling of prevented colorectal cancer cases. NEJM, 384
    4. Stanford Medicine. (2022). Multi-center validation of deep learning for radiograph interpretation. Radiology, 302(1)
    5. Nuance Communications. (2023). DAX system implementation outcomes. Health Affairs, 42(1)
    6. PathAI. (2022). Pathologist adoption of explainable AI systems. Modern Pathology, 35
    7. MELLODDY Consortium. (2022). Federated learning for pharmaceutical research. Nature Machine Intelligence, 4
    8. Massachusetts General Hospital. (2021). Clinical workflow integration failure points for AI. JAMIA, 28(9)
    9. Cleveland Clinic. (2023). AI Integration Framework outcomes. Healthcare Innovation, 11(2)
    10. American Medical Association. (2022). Physician AI literacy assessment. Journal of Medical Education, 97(6)
    11. Centers for Medicare & Medicaid Services. (2023). Healthcare AI economic impact analysis
    12. FDA. (2023). Proposed framework for AI/ML-based SaMD. Regulatory Science Forum
    """,
    """
    # Quantum Computing Applications in Pharmaceutical Discovery: Capabilities, Limitations, and Industry Transformation
    
    ## Executive Summary
    
    This analysis evaluates the integration of quantum computing technologies into pharmaceutical R&D workflows, examining current capabilities, near-term applications, and long-term industry transformation potential. Based on benchmarking across 17 pharmaceutical companies and 8 quantum technology providers, we provide a comprehensive assessment of this emerging computational paradigm and its implications for drug discovery economics.
    
    ## Current Quantum Computing Capabilities
    
    ### Hardware Platforms Assessment
    
    **Superconducting quantum processors** (IBM, Google, Rigetti) currently provide the most mature platform with IBM's 433-qubit Osprey system demonstrating quantum volume of 128 and error rates approaching 10^-3 per gate operation. While impressive relative to 2018 benchmarks, these systems remain limited by coherence times (averaging 114 microseconds) and require operating temperatures near absolute zero, creating significant infrastructure requirements.
    
    **Trapped-ion quantum computers** (IonQ, Quantinuum) offer superior coherence times exceeding 10 seconds and all-to-all connectivity but operate at slower gate speeds. IonQ's 32-qubit system achieved algorithmic qubits (#AQ) of 20, setting a record for effective computational capability when error mitigation is considered. Quantinuum's H-Series demonstrated the first logical qubit with real-time quantum error correction, a significant milestone towards fault-tolerant quantum computing.
    
    **Photonic quantum systems** (Xanadu, PsiQuantum) represent an alternative approach with potentially simpler scaling requirements. Xanadu's Borealis processor demonstrated quantum advantage for specific sampling problems but lacks the gate-based universality required for most pharmaceutical applications. PsiQuantum's fault-tolerant silicon photonic approach continues rapid development with semiconductor manufacturing partner GlobalFoundries but remains pre-commercial.
    
    **Neutral atom platforms** (QuEra, Pasqal) entered commercial accessibility in 2023, offering unprecedented qubit counts (QuEra: 256 atoms) with programmable geometries particularly suited for quantum simulation of molecular systems. Recent demonstrations of 3D atom arrangements provide promising avenues for simulating protein-ligand interactions.
    
    ### Quantum Algorithm Development
    
    Pharmaceutical applications currently focus on three quantum algorithm classes:
    
    1. **Variational Quantum Eigensolver (VQE)** algorithms have progressed significantly for molecular ground state energy calculations, with Riverlane's enhanced VQE implementations demonstrating accuracy within 1.5 kcal/mol for molecules up to 20 atoms on IBM's 127-qubit processors. Merck's collaboration with Zapata Computing improved convergence rates by 300% through adaptive ansatz methods.
    
    2. **Quantum Machine Learning (QML)** approaches for binding affinity prediction have shown mixed results. Pfizer's implementation of quantum convolutional neural networks (QCNN) demonstrated a 22% improvement in binding affinity predictions for their kinase inhibitor library, while AstraZeneca's quantum support vector machine approach showed no significant advantage over classical methods for their dataset.
    
    3. **Quantum Annealing** for conformational search remains dominated by D-Wave's 5,000+ qubit systems, with Boehringer Ingelheim reporting successful applications in peptide folding predictions. However, comparisons with enhanced classical methods (particularly those using modern GPUs) show quantum advantage remains elusive for most production cases.
    
    ## Pharmaceutical Applications Landscape
    
    ### Virtual Screening Transformation
    
    GSK's quantum computing team achieved a significant milestone in 2022 through quantum-classical hybrid algorithms that accelerated screening of 10^7 compounds against novel SARS-CoV-2 targets. Their approach used classical computers for initial filtering followed by quantum evaluation of 10^4 promising candidates, identifying 12 compounds with nanomolar binding affinities subsequently confirmed by experimental assays. While impressive, the computational requirements exceeded $1.2M and required specialized expertise from partners at Quantinuum.
    
    ### Molecular Property Prediction
    
    Roche's collaboration with Cambridge Quantum Computing (now Quantinuum) demonstrated quantum advantage for dipole moment calculations in drug-like molecules, achieving accuracy improvements of 16% compared to density functional theory methods while potentially offering asymptotic speedup as molecule size increases. Their hybrid quantum-classical approach requires significantly fewer qubits than full quantum simulation, making it commercially relevant within the NISQ (Noisy Intermediate-Scale Quantum) era of hardware.
    
    ### Retrosynthesis Planning
    
    Quantum approaches to synthetic route planning remain largely theoretical with limited experimental validation. MIT-Takeda research demonstrated proof-of-concept for mapping retrosynthesis to quantum walks on Johnson graphs, with preliminary results showing promise for identifying non-obvious synthetic pathways. Commercial application appears distant (5-8 years) given current hardware limitations.
    
    ## Economic Implications Analysis
    
    Our economic model quantifies four significant impacts on pharmaceutical R&D:
    
    1. **Preclinical timeline compression**: Currently estimated at 2-5% (0.5-1.3 months) but projected to reach 15-30% by 2030 as quantum hardware capabilities expand, potentially reducing time-to-market by up to 9 months for novel compounds
    
    2. **Candidate quality improvements**: Quantum-enhanced binding affinity and ADMET property predictions demonstrate 7-18% higher success rates in early clinical phases across our analysis of 87 compounds that utilized quantum computational methods in preclinical development
    
    3. **Novel mechanism identification**: Quantum simulation of previously intractable biological targets (particularly intrinsically disordered proteins and complex protein-protein interactions) could expand the druggable proteome by an estimated 8-14% according to our analysis of protein data bank targets
    
    4. **R&D productivity impacts**: A 10% improvement in candidate quality translates to approximately $310M in reduced clinical development costs per approved drug by reducing late-stage failures
    
    ## Investment and Adoption Patterns
    
    Pharmaceutical quantum computing investment has accelerated dramatically, with cumulative industry investment growing from $18M (2018) to $597M (2023). Investment strategies fall into three categories:
    
    1. **Direct infrastructure investment** (Roche, Merck): Building internal quantum teams and securing dedicated quantum hardware access
    
    2. **Collaborative research partnerships** (GSK, Biogen, Novartis): Forming multi-year academic and commercial partnerships focused on specific computational challenges
    
    3. **Quantum-as-a-service utilization** (Majority approach): Accessing quantum capabilities through cloud providers with limited internal expertise development
    
    Our analysis of 23 pharmaceutical companies indicates:
    - 19% have established dedicated quantum computing teams
    - 43% have active research collaborations with quantum providers
    - 78% report evaluating quantum capabilities for specific workflows
    - 100% express concerns about quantum talent acquisition challenges
    
    ## Future Outlook and Strategic Recommendations
    
    The pharmaceutical quantum computing landscape will evolve through three distinct phases:
    
    **Near-term (1-3 years)**: Hybrid quantum-classical algorithms will demonstrate incremental value in specific niches, particularly molecular property calculations and conformational analysis of small to medium-sized molecules. Successful organizations will combine quantum capabilities with enhanced classical methods rather than seeking immediate quantum advantage.
    
    **Mid-term (3-7 years)**: Error-corrected logical qubits will enable more robust quantum chemistry applications with demonstrable advantage for drug discovery workflows. Companies with established quantum capabilities will gain first-mover advantages in applying these technologies to proprietary chemical matter.
    
    **Long-term (7+ years)**: Fault-tolerant quantum computers with thousands of logical qubits could transform pharmaceutical R&D by enabling full quantum mechanical simulation of protein-drug interactions and previously intractable biological systems. This capability could fundamentally alter drug discovery economics by dramatically reducing empirical screening requirements.
    
    ## References
    
    1. IBM Quantum. (2023). Osprey processor technical specifications and benchmarking
    2. IonQ. (2023). Algorithmic qubit benchmarking methodology and results
    3. Quantinuum. (2022). H-Series logical qubit demonstration
    4. Xanadu. (2022). Borealis quantum advantage results. Nature Physics, 18
    5. QuEra. (2023). Neutral atom quantum processor capabilities. Science, 377
    6. Riverlane & Merck. (2022). Enhanced VQE implementations for molecular ground state calculations
    7. Pfizer Quantum Team. (2023). QCNN for binding affinity prediction. Journal of Chemical Information and Modeling
    8. AstraZeneca. (2022). Comparative analysis of quantum and classical ML methods
    9. Boehringer Ingelheim. (2023). Quantum annealing for peptide conformational search
    10. GSK Quantum Computing Team. (2022). Quantum-classical hybrid screening against SARS-CoV-2
    11. Roche & Cambridge Quantum Computing. (2023). Quantum advantage for dipole moment calculations
    12. MIT-Takeda Quantum Research. (2022). Mapping retrosynthesis to quantum walks
    13. PhRMA Quantum Computing Working Group. (2023). Pharmaceutical R&D impact analysis
    """,
    """
    # Neuroplasticity in Cognitive Rehabilitation: Mechanisms, Interventions, and Clinical Applications
    
    ## Abstract
    
    This multidisciplinary review synthesizes current understanding of neuroplasticity mechanisms underlying cognitive rehabilitation, evaluating intervention efficacies across five domains of cognitive function following acquired brain injury. Integrating data from 142 clinical studies with advanced neuroimaging findings, we present evidence-based recommendations for clinical practice and identify promising emerging approaches.
    
    ## Neurobiological Foundations of Rehabilitation-Induced Plasticity
    
    ### Cellular and Molecular Mechanisms
    
    Recent advances in understanding activity-dependent plasticity have revolutionized rehabilitation approaches. The pioneering work of Dr. Alvarez-Buylla at UCSF has demonstrated that even the adult human brain maintains neurogenic capabilities in the hippocampus and subventricular zone, with newly generated neurons integrating into existing neural circuits following injury. Transcriptomic studies by Zhang et al. (2021) identified 37 genes significantly upregulated during rehabilitation-induced recovery, with brain-derived neurotrophic factor (BDNF) and insulin-like growth factor-1 (IGF-1) showing particularly strong associations with positive outcomes.
    
    Post-injury plasticity occurs through multiple complementary mechanisms:
    
    1. **Synaptic remodeling**: Two-photon microscopy studies in animal models reveal extensive dendritic spine turnover within peri-lesional cortex during the first 3-8 weeks post-injury. The pioneering work of Professor Li-Huei Tsai demonstrates that enriched rehabilitation environments increase spine formation rates by 47-68% compared to standard housing conditions.
    
    2. **Network reorganization**: Professor Nicholas Schiff's research at Weill Cornell demonstrates that dormant neural pathways can be functionally recruited following injury through targeted stimulation. Their multimodal imaging studies identified specific thalamocortical circuits that, when engaged through non-invasive stimulation, facilitated motor recovery in 72% of chronic stroke patients previously classified as "plateaued."
    
    3. **Myelination dynamics**: Recent discoveries by Dr. Fields at NIH demonstrate activity-dependent myelination as a previously unrecognized form of neuroplasticity. Diffusion tensor imaging studies by Wang et al. (2022) show significant increases in white matter integrity following intensive cognitive training, correlating with functional improvements (r=0.62, p<0.001).
    
    ### Neuroimaging Correlates of Successful Rehabilitation
    
    Longitudinal multimodal neuroimaging studies have identified several biomarkers of successful cognitive rehabilitation:
    
    - **Functional connectivity reorganization**: Using resting-state fMRI, Northoff's laboratory documented that successful attention training in 67 TBI patients correlated with increased connectivity between the dorsolateral prefrontal cortex and posterior parietal regions (change in z-score: 0.43 ± 0.12), while unsuccessful cases showed no significant connectivity changes.
    
    - **Cortical thickness preservation**: Dr. Gabrieli's team at MIT found that cognitive rehabilitation initiated within 30 days of injury preserved cortical thickness in vulnerable regions, with each week of delay associated with 0.8% additional atrophy in domain-relevant cortical regions.
    
    - **Default mode network modulation**: Advanced network analyses by Dr. Marcus Raichle demonstrate that cognitive rehabilitation success correlates with restoration of appropriate task-related deactivation of the default mode network, suggesting intervention effectiveness can be monitored through this biomarker.
    
    ## Evidence-Based Intervention Analysis
    
    ### Attention and Executive Function Rehabilitation
    
    Our meta-analysis of 42 randomized controlled trials evaluating attention training programs reveals three intervention approaches with significant effect sizes:
    
    1. **Adaptive computerized training** (Hedges' g = 0.68, 95% CI: 0.54-0.82): Programs like Attention Process Training showed transfer to untrained measures when training adapts in real-time to performance. The NYU-Columbia adaptive attention protocol demonstrated maintenance of gains at 18-month follow-up (retention rate: 83%).
    
    2. **Metacognitive strategy training** (Hedges' g = 0.57, 95% CI: 0.41-0.73): The Toronto Hospital's Strategic Training for Executive Control program resulted in significant improvements on ecological measures of executive function. Moderator analyses indicate effectiveness increases when combined with daily strategy implementation exercises (interaction effect: p=0.002).
    
    3. **Neurostimulation-enhanced approaches**: Combined tDCS-cognitive training protocols developed at Harvard demonstrate 37% greater improvement compared to cognitive training alone. Targeting the right inferior frontal gyrus with 2mA anodal stimulation during inhibitory control training shows particular promise for impulsivity reduction (Cohen's d = 0.74).
    
    ### Memory Rehabilitation Approaches
    
    Memory intervention effectiveness varies substantially by memory system affected and etiology:
    
    - **Episodic memory**: For medial temporal lobe damage, compensatory approaches using spaced retrieval and errorless learning demonstrate the strongest evidence. Dr. Schacter's laboratory protocol combining elaborative encoding with distributed practice shows a remarkable 247% improvement in functional memory measures compared to intensive rehearsal techniques.
    
    - **Prospective memory**: Implementation intention protocols developed by Professor Gollwitzer show transfer to daily functioning with large effect sizes (d = 0.92) when combined with environmental restructuring. Smartphone-based reminder systems increased medication adherence by 43% in our 12-month community implementation study.
    
    - **Working memory**: Recent controversy surrounding n-back training was addressed in Professor Klingberg's definitive multi-site study demonstrating domain-specific transfer effects. Their adaptive protocol produced sustainable working memory improvements (40% above baseline at 6-month follow-up) when training exceeded 20 hours and incorporated gradually increasing interference control demands.
    
    ## Clinical Application Framework
    
    ### Precision Rehabilitation Medicine Approach
    
    Our analysis indicates rehabilitation effectiveness increases substantially when protocols are tailored using a precision medicine framework:
    
    1. **Comprehensive neurocognitive phenotyping**: The McGill Cognitive Rehabilitation Battery enables identification of specific processing deficits, allowing intervention targeting. Machine learning analysis of 1,247 patient profiles identified 11 distinct neurocognitive phenotypes that respond differentially to specific interventions.
    
    2. **Biomarker-guided protocol selection**: EEG connectivity measures predicted response to attention training with 76% accuracy in our validation cohort, potentially reducing non-response rates. Professor Knight's laboratory demonstrated that P300 latency specifically predicts processing speed training response (AUC = 0.81).
    
    3. **Adaptive progression algorithms**: Real-time difficulty adjustment based on multiple performance parameters rather than accuracy alone increased transfer effects by 34% compared to standard adaptive approaches. The computational model developed by Stanford's Poldrack laboratory dynamically optimizes challenge levels to maintain engagement while maximizing error-based learning.
    
    ### Implementation Science Considerations
    
    Our implementation analysis across 24 rehabilitation facilities identified critical factors for successful cognitive rehabilitation programs:
    
    - **Rehabilitation intensity and timing**: Early intervention (< 6 weeks post-injury) with high intensity (minimum 15 hours/week of direct treatment) demonstrated superior outcomes (NNT = 3.2 for clinically significant improvement).
    
    - **Therapist expertise effects**: Specialized certification in cognitive rehabilitation was associated with 28% larger treatment effects compared to general rehabilitation credentials.
    
    - **Technology augmentation**: Hybrid models combining therapist-directed sessions with home-based digital practice demonstrated optimal cost-effectiveness (ICER = $12,430/QALY) while addressing access barriers.
    
    ## Future Directions and Emerging Approaches
    
    Several innovative approaches show promise for enhancing neuroplasticity during cognitive rehabilitation:
    
    1. **Closed-loop neurostimulation**: Dr. Suthana's team at UCLA demonstrated that theta-burst stimulation delivered precisely during specific phases of hippocampal activity enhanced associative memory formation by 37% in patients with mild cognitive impairment.
    
    2. **Pharmacologically augmented rehabilitation**: The RESTORE trial combining daily atomoxetine with executive function training demonstrated synergistic effects (interaction p<0.001) compared to either intervention alone. Professor Feeney's research suggests a critical 30-minute window where noradrenergic enhancement specifically promotes task-relevant plasticity.
    
    3. **Virtual reality cognitive training**: Immersive VR protocols developed at ETH Zurich demonstrated transfer to real-world functioning by simulating ecologically relevant scenarios with graduated difficulty. Their randomized trial showed 3.2× greater functional improvement compared to matched non-immersive training.
    
    4. **Sleep optimization protocols**: The Northwestern sleep-enhanced memory consolidation protocol increased rehabilitation effectiveness by 41% by delivering targeted memory reactivation during slow-wave sleep, suggesting rehabilitation schedules should specifically incorporate sleep architecture considerations.
    
    ## Conclusion
    
    Cognitive rehabilitation effectiveness has improved substantially through integration of neuroplasticity principles, advanced technology, and precision intervention approaches. Optimal outcomes occur when interventions target specific neurocognitive mechanisms with sufficient intensity and are tailored to individual patient profiles. Emerging approaches leveraging closed-loop neurotechnology and multimodal enhancement strategies represent promising directions for further advancing rehabilitation outcomes.
    
    ## References
    
    1. Alvarez-Buylla, A., & Lim, D. A. (2022). Neurogenesis in the adult human brain following injury
    2. Zhang, Y., et al. (2021). Transcriptomic analysis of rehabilitation-responsive genes
    3. Tsai, L. H., et al. (2023). Environmental enrichment effects on dendritic spine dynamics
    4. Schiff, N. D. (2022). Recruitment of dormant neural pathways following brain injury
    5. Fields, R. D. (2021). Activity-dependent myelination as a form of neuroplasticity
    6. Wang, X., et al. (2022). White matter integrity changes following cognitive training
    7. Northoff, G., et al. (2023). Functional connectivity reorganization during attention training
    8. Gabrieli, J. D., et al. (2021). Relationship between intervention timing and cortical preservation
    9. Raichle, M. E. (2022). Default mode network dynamics as a biomarker of rehabilitation efficacy
    10. NYU-Columbia Collaborative. (2023). Adaptive attention protocol long-term outcomes
    11. Schacter, D. L., et al. (2021). Elaborative encoding with distributed practice for episodic memory
    12. Gollwitzer, P. M., & Oettingen, G. (2022). Implementation intentions for prospective memory
    13. Klingberg, T., et al. (2023). Multi-site study of adaptive working memory training
    14. Poldrack, R. A., et al. (2022). Computational models for optimizing learning parameters
    15. Suthana, N., et al. (2023). Phase-specific closed-loop stimulation for memory enhancement
    16. Feeney, D. M., & Sutton, R. L. (2022). Pharmacological enhancement of rehabilitation
    17. ETH Zurich Rehabilitation Engineering Group. (2023). Virtual reality cognitive training
    18. Northwestern Memory & Cognition Laboratory. (2022). Sleep-enhanced memory consolidation
    """,
]


async def display_workflow_diagram(workflow):
    """Display a visual representation of the workflow DAG."""
    console.print("\n[bold cyan]Workflow Execution Plan[/bold cyan]")

    # Create a tree representation of the workflow
    tree = Tree("[bold yellow]Research Analysis Workflow[/bold yellow]")

    # Track dependencies for visualization
    dependencies = {}
    for stage in workflow:
        stage_id = stage["stage_id"]
        deps = stage.get("depends_on", [])
        for dep in deps:
            if dep not in dependencies:
                dependencies[dep] = []
            dependencies[dep].append(stage_id)

    # Add stages without dependencies first (roots)
    root_stages = [s for s in workflow if not s.get("depends_on")]
    stage_map = {s["stage_id"]: s for s in workflow}

    def add_stage_to_tree(parent_tree, stage_id):
        stage = stage_map[stage_id]
        tool = stage["tool_name"]
        node_text = f"[bold green]{stage_id}[/bold green] ([cyan]{tool}[/cyan])"

        if "iterate_on" in stage:
            node_text += " [italic](iterative)[/italic]"

        stage_node = parent_tree.add(node_text)

        # Add children (stages that depend on this one)
        children = dependencies.get(stage_id, [])
        for child in children:
            add_stage_to_tree(stage_node, child)

    # Build the tree
    for root in root_stages:
        add_stage_to_tree(tree, root["stage_id"])

    # Print the tree
    console.print(tree)

    # Display additional workflow statistics
    table = Table(title="Workflow Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Stages", str(len(workflow)))
    table.add_row("Parallel Stages", str(len(root_stages)))
    table.add_row("Iterative Stages", str(sum(1 for s in workflow if "iterate_on" in s)))

    console.print(table)


async def display_execution_progress(workflow_future):
    """Display a live progress indicator while the workflow executes."""
    with Progress(
        SpinnerColumn(), TextColumn("[bold blue]{task.description}"), console=console
    ) as progress:
        task = progress.add_task("[yellow]Executing workflow...", total=None)
        result = await workflow_future
        progress.update(task, completed=True, description="[green]Workflow completed!")
        return result


async def visualize_results(results):
    """Create visualizations of the workflow results."""
    console.print("\n[bold magenta]Research Analysis Results[/bold magenta]")

    # Set up layout
    layout = Layout()
    layout.split_column(
        Layout(name="header"),
        Layout(name="statistics"),
        Layout(name="summaries"),
        Layout(name="extracted_entities"),
    )

    # Header
    layout["header"].update(Panel("[bold]Advanced Research Assistant Results[/bold]", style="blue"))

    # Statistics
    stats_table = Table(title="Document Processing Statistics")
    stats_table.add_column("Document", style="cyan")
    stats_table.add_column("Word Count", style="green")
    stats_table.add_column("Entity Count", style="yellow")

    try:
        chunking_result = results["results"]["chunking_stage"]["output"]
        entity_results = results["results"]["entity_extraction_stage"]["output"]

        for i, doc_stats in enumerate(chunking_result.get("document_stats", [])):
            entity_count = len(entity_results[i].get("entities", []))
            stats_table.add_row(
                f"Document {i + 1}", str(doc_stats.get("word_count", "N/A")), str(entity_count)
            )
    except (KeyError, IndexError) as e:
        console.print(f"[red]Error displaying statistics: {e}[/red]")

    layout["statistics"].update(stats_table)

    # Summaries
    summary_panels = []
    try:
        summaries = results["results"]["summary_stage"]["output"]
        for i, summary in enumerate(summaries):
            summary_panels.append(
                Panel(
                    summary.get("summary", "No summary available"),
                    title=f"Document {i + 1} Summary",
                    border_style="green",
                )
            )
    except (KeyError, IndexError) as e:
        summary_panels.append(
            Panel(f"Error retrieving summaries: {e}", title="Summary Error", border_style="red")
        )

    layout["summaries"].update(summary_panels)

    # Extracted entities
    try:
        final_analysis = results["results"]["final_analysis_stage"]["output"]
        json_str = Syntax(
            str(final_analysis.get("analysis", "No analysis available")),
            "json",
            theme="monokai",
            line_numbers=True,
        )
        layout["extracted_entities"].update(
            Panel(json_str, title="Final Analysis", border_style="magenta")
        )
    except (KeyError, IndexError) as e:
        layout["extracted_entities"].update(
            Panel(
                f"Error retrieving final analysis: {e}", title="Analysis Error", border_style="red"
            )
        )

    # Print layout
    console.print(layout)

    # Display execution time
    console.print(
        f"\n[bold green]Total workflow execution time:[/bold green] "
        f"{results.get('total_processing_time', 0):.2f} seconds"
    )


def create_research_workflow():
    """Define a complex research workflow with multiple parallel and sequential stages."""
    workflow = [
        # Initial document processing stages (run in parallel for all documents)
        {
            "stage_id": "chunking_stage",
            "tool_name": "chunk_document",
            "params": {"text": "${documents}", "chunk_size": 1000, "get_stats": True},
        },
        # Entity extraction runs in parallel with summarization
        {
            "stage_id": "entity_extraction_stage",
            "tool_name": "extract_entity_graph",
            "params": {
                "text": "${documents}",
                "entity_types": ["organization", "person", "concept", "location", "technology"],
                "include_relations": True,
                "confidence_threshold": 0.7,
            },
        },
        # Summarization stage (iterate over each document)
        {
            "stage_id": "summary_stage",
            "tool_name": "summarize_document",
            "params": {
                "text": "${documents}",
                "max_length": 150,
                "focus_on": "key findings and implications",
            },
        },
        # Classification of document topics
        {
            "stage_id": "classification_stage",
            "tool_name": "text_classification",
            "depends_on": ["chunking_stage"],
            "params": {
                "text": "${chunking_stage.document_text}",
                "categories": [
                    "Climate & Environment",
                    "Technology",
                    "Healthcare",
                    "Economy",
                    "Social Policy",
                    "Scientific Research",
                ],
                "provider": Provider.OPENAI.value,
                "multi_label": True,
                "confidence_threshold": 0.6,
            },
        },
        # Generate structured insights from entity analysis
        {
            "stage_id": "entity_insights_stage",
            "tool_name": "extract_json",
            "depends_on": ["entity_extraction_stage"],
            "params": {
                "text": "${entity_extraction_stage.text_output}",
                "schema": {
                    "key_entities": "array",
                    "primary_relationships": "array",
                    "research_domains": "array",
                },
                "include_reasoning": True,
            },
        },
        # Cost-optimized final analysis
        {
            "stage_id": "model_selection_stage",
            "tool_name": "recommend_model",
            "depends_on": ["summary_stage", "classification_stage", "entity_insights_stage"],
            "params": {
                "task_type": "complex analysis and synthesis",
                "expected_input_length": 3000,
                "expected_output_length": 1000,
                "required_capabilities": ["reasoning", "knowledge"],
                "priority": "balanced",
            },
        },
        # Final analysis and synthesis
        {
            "stage_id": "final_analysis_stage",
            "tool_name": "chat_completion",
            "depends_on": [
                "model_selection_stage",
                "summary_stage",
                "classification_stage",
                "entity_insights_stage",
            ],
            "params": {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a research assistant synthesizing information from multiple documents.",
                    },
                    {
                        "role": "user",
                        "content": "Analyze the following research summaries, classifications, and entity insights. Provide a comprehensive analysis that identifies cross-document patterns, contradictions, and key insights. Format the response as structured JSON.\n\nSummaries: ${summary_stage.summary}\n\nClassifications: ${classification_stage.classifications}\n\nEntity Insights: ${entity_insights_stage.content}",
                    },
                ],
                "model": "${model_selection_stage.recommendations[0].model}",
                "response_format": {"type": "json_object"},
            },
        },
    ]

    return workflow


async def main():
    """Run the complete research assistant workflow demo."""
    console.print(Rule("[bold magenta]Advanced Research Workflow Demo[/bold magenta]"))
    tracker = CostTracker()  # Instantiate tracker

    try:
        # Display header
        console.print(
            Panel.fit(
                "[bold cyan]Advanced Research Assistant Workflow Demo[/bold cyan]\n"
                "Powered by NetworkX DAG-based Workflow Engine",
                title="Ultimate MCP Server",
                border_style="green",
            )
        )

        # Create the workflow definition
        workflow = create_research_workflow()

        # Visualize the workflow before execution
        await display_workflow_diagram(workflow)

        # Prompt user to continue
        console.print("\n[yellow]Press Enter to execute the workflow...[/yellow]", end="")
        input()

        # Execute workflow with progress display
        workflow_future = execute_optimized_workflow(
            documents=SAMPLE_DOCS, workflow=workflow, max_concurrency=3
        )

        results = await display_execution_progress(workflow_future)

        # Track cost if possible
        if results and isinstance(results, dict) and "cost" in results:
            try:
                total_cost = results.get("cost", {}).get("total_cost", 0.0)
                processing_time = results.get("total_processing_time", 0.0)
                # Provider/Model is ambiguous here, use a placeholder
                trackable = TrackableResult(
                    cost=total_cost,
                    input_tokens=0,  # Not aggregated
                    output_tokens=0,  # Not aggregated
                    provider="workflow",
                    model="research_workflow",
                    processing_time=processing_time,
                )
                tracker.add_call(trackable)
            except Exception as track_err:
                logger.warning(f"Could not track workflow cost: {track_err}", exc_info=False)

        if results:
            console.print(Rule("[bold green]Workflow Execution Completed[/bold green]"))
            await visualize_results(results.get("outputs", {}))
        else:
            console.print("[bold red]Workflow execution failed or timed out.[/bold red]")

    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")

    # Display cost summary
    tracker.display_summary(console)


if __name__ == "__main__":
    asyncio.run(main())
