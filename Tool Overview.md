We brought together various data sources to generate cluster recommendations for projects currently in the CAISO Interconnection Queue. Both supervised and unsupervised machine learning techniques were used to build the scoring mechanism and resulting clusters. 

This short video outlines the basic workflow: Set parameter weights, select a base project for the cluster, and review the results.

https://github.com/haschuele/Overpowered/assets/107952982/7a9f53d7-042f-409a-ae10-1ad10654a046

The scoring algorithm uses 4 interpretable parameters (below). Users can set custom weights for these parameters based on the unique needs of their ISO.

- Likelihood of Approval: The likelihood that a given project will succeed, independent of the rest of the cluster, based on past project applications.
- Location: The geospatial proximity between two projects.
- Process: This summarizes the readiness of each project. It includes operational variables, such as the project’s position in the Queue, the date it's expected to go online, and its permit status. We want to discourage “line skipping” by grouping projects that are closer together in the Queue. We also want to encourage ease of construction and real-life operations by having projects in the same geography go online at similar times.
- Infrastructure: The similarity of the project build types. For example, two solar projects can be studied under the same set of assumptions, which is more efficient than two projects of different types.

In the results section, users can dig into the custom scoring mechanism to understand the strength of each cluster. Metrics for the entire cluster at located at the top, such as "Net Transmission Capacity" which measures the lack of infrastructure given existing constraints and the cluster's proposed new MWs to the grid. The table includes individual metrics for each project, followed by an "Overall" similarity score to the base project.

![Sample Results](https://github.com/haschuele/Overpowered/blob/main/Sample%20Results.png)




The tool also provides maps for visualization...
