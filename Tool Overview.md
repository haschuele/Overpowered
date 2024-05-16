# The Tool
We brought together various data sources to generate cluster recommendations for projects currently in the CAISO Interconnection Queue. Both supervised and unsupervised machine learning techniques were used to build the scoring mechanism and resulting clusters. 

This short video outlines the basic workflow: Set parameter weights based on the unique needs of the ISO, select a base project from the existing CAISO Queue, and review the recommended cluster results.

https://github.com/haschuele/Overpowered/assets/107952982/7a9f53d7-042f-409a-ae10-1ad10654a046

The scoring algorithm uses 4 interpretable parameters:

- Likelihood of Approval: The likelihood that a given project will succeed, independent of the rest of the cluster, based on past project applications.
- Location: The geospatial proximity between two projects.
- Process: This summarizes the readiness of each project. It includes operational variables, such as the project’s position in the Queue, the date it's expected to go online, and its permit status. We want to discourage “line skipping” by grouping projects that are closer together in the Queue. We also want to encourage ease of construction and real-life operations by having projects in the same geography go online at similar times.
- Infrastructure: The similarity of the project build types. For example, two solar projects can be studied under the same set of assumptions, which is more efficient than two projects of different types.

In the results section, users can dig into the custom scoring mechanism to understand the strength of each cluster. Metrics for the entire cluster at located at the top, such as "Net Transmission Capacity" which measures the infrastructure needed to accomodate the cluster's proposed MWs to grid given existing transmission constraints. The table includes individual metrics for each project, the last column of which is an "Overall" similarity score to the base project.

![Sample Results](https://github.com/haschuele/Overpowered/blob/main/Sample%20Results.png)

A map of the projects in the recommended cluster can be seen next to the results.

![Tule Recommended Cluster](https://github.com/haschuele/Overpowered/blob/main/Tule%20Recommended%20Cluster.png)

The Overpowered tool also contains customizable maps so users can visualize the infrastructure that went into the scoring mechanism, including existing substations, retired power plants, current queue projects, and planned infrastructure. 

![Interactive Map](https://github.com/haschuele/Overpowered/blob/main/Interactive%20Map.png)

# The Value
We had to overcome a series of technical challenges during this project, such as piecing together various data sources of different types and calculating realistic values for unknown variables. However, the biggest challenges were those related to human decision-making and allowing for flexibility in an inexact problem space. We conducted extensive research and a series of subject-matter-expert interviews to determine how to abstract the technical machine learning and human-centric grid process considerations in a way that would allow anyone to be able to successsfully interact with our tool.

The mission of this project was to create a more efficient way of connecting new energy sources to the power grid. Our solution speeds up the Queue, reduces developer dropout, allows for flexibility, and provides visibility into the results. With these improvements, we take another step towards a green power grid, faster.
