# How to Contribute to FedML

[FedML](https://github.com/FedML-AI/FedML) is an open and community-driven project. Everyone is welcome to contribute!


## Interacting with the FedML Community

- [GitHub issues tracker](https://github.com/FedML-AI/FedML/issues): addressing and discussing functional issues and problems of the FedML library. 
- [Slack channel #general](https://fedml.slack.com/archives/C0172MH1PRV): open-ended questions and issues related to the FedML open-source library.
- [Slack channel #fedml-system](https://fedml.slack.com/archives/C01D33QU6PN): open-ended questions and issues related to system errors and issues.
- [Slack channel #fedml-model-and-dataset](https://fedml.slack.com/archives/C018797FC7J): open-ended questions and issues related to model definitions and (customized) dataset loaders.
- [Slack channel #fedml-algorithms](https://fedml.slack.com/archives/C017G2YQGFP): open-ended questions and issues related to algorithmic optimizations and implementations.
- [Discord](https://discord.gg/VcEBxSKh): open-ended questions and issues related to the FedML open-source library, similar to Slack.


## Submitting a Bug Report / Feature Request / General Q&A

Before submitting a github issue or a feature request, ensure the issue has not been already reported under
[Issues](https://github.com/FedML-AI/FedML/issues), either in the [open](https://github.com/FedML-AI/FedML/issues?q=is%3Aopen+is%3Aissue) or [closed](https://github.com/FedML-AI/FedML/issues?q=is%3Aissue+is%3Aclosed) issues list 
and it is not currently being addressed by other [pull requests](https://github.com/FedML-AI/FedML/pulls). For questions that required to be asnwered in a timely manner it is recommended to ask the [Slack community](https://fedml.slack.com/archives/C0172MH1PRV).

If you're unable to find an open issue addressing your problem, then you can [open a new one](https://github.com/FedML-AI/FedML/issues/new). Please make sure to include: 
- **Issue Title**: a consice explanation of the issue 
- **Description**: be as descriptive as possible so that the issue can be reproduced by others. The description needs to contain the following checklist:
    - [x] your environment's fedml library and environment
        - **hint:** run `fedml env`
    - [x] your environment's operating system specs
    - [x] your environment's hardware specs
    - [x] the sequence of execution commands creating this issue
    - [x] source code to reproduce the error wherever necessary
- **Label**: need to select at least one label from the following list:
    - `bug`: something is not working at all
    - `documentation`: improvements or additions to documentation
    - `good first issue`: if this is the first issue of the user
    - `help wanted`: if this is an advance feature request and critical help is needed
    - `hotfix`: a small change to a speficic functionality is needed
    - `question`: if the issue is an implementation or other open-ended question
    - `research-projects`: if the issue is related to a research project/question
    - `TODO features`: if the issue requests the implementation of new feature


## Contributing Code - Opening a PR

Before you open a new pull request or contribute code, please make sure you have gone over the documentation, have successfully installed the FedML library and you are able to run a working example:
- [FedML Installation Docs](https://doc.fedml.ai/open-source/installation)
- [FedML Examples](https://github.com/FedML-AI/FedML/tree/master/python/examples)

However, to avoid duplicating work, it is highly recommended before a new PR is opened to search through the
[issue tracker](https://github.com/FedML-AI/FedML/issues) and
[pull requests list](https://github.com/FedML-AI/FedML/pulls). One easy way to scan over existing issues and PRs, is by searching using one of the following labels 
(`bug, help wanted, hotfix, TODO features`), for instance for an existing `TODO features` [you can look here](https://github.com/FedML-AI/FedML/labels/TODO%20features).

The procedure to start working on a new PR is:
1. checkout the [development branch](https://github.com/FedML-AI/FedML/tree/dev/v0.7.0)
2. make any necessary changes locally and push to a new branch following the convention `<username>/<PR_name>`
    - `username` is the name of your GitHub account and `PR_name` is a succint name describing your PR
3.  once you have pushed your PR you need to request a merge with the development branch

Once the PR is approved, it will be merged into `dev/v0.7.0` and subsequently to the `main` branch in the next [FedML Release](https://github.com/FedML-AI/FedML/releases).

