This project brought together various data sources, including the CAISO Interconnection Queue, to create a custom scoring mechanism which provides insight into the strength of each recommended cluster. 

Both supervised and unsupervised machine learning techniques were used to build the scoring mechanism and resulting clusters. The score  and is comprised of 4 interpretable categories (below). Users can set custom weights for these categories based on the unique needs of their ISO.

- Likelihood of Approval: This is the likelihood that a given project would succeed independent of the rest of the cluster, based on past project applications.
- Location: This measures the geospatial proximity between two projects.
- Process: This summarizes the readiness of each project. It includes operational variables, such as the project’s position in the Queue, the date it's expected to go online, and its permit status. We want to discourage “line skipping” by grouping projects that are closer together in the Queue. We also want to encourage ease of construction and real-life operations by having projects in the same geography go online at similar times.
- Infrastructure: This captures the similarity of the project build types. For example, two solar projects can be studied under the same set of assumptions, which is more efficient than two projects of different types.

Once weights and a base project have been selected, the tool will provide a recommended cluster and visibility into the supporting scores.

![Sample Results](https://github.com/haschuele/Overpowered/blob/main/Sample%20Results.png)

The tool also provides maps for visualization...
