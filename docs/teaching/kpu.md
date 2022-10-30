---
tags:
  - curriculum design
comments: true

---

# CSIT@KPU

Some students at CSIT department of KPU found it a bit hard to navigate the curriculum, so I spend some time to build the flowchart below based on public data on university websites. Note that this is not guarantteed to be the lastest and correct version. Please leave commments if you notice any mistakes. Hope it may help.

<iframe width="1400" height="750" src="https://miro.com/app/live-embed/uXjVOGRVlQc=/?moveToViewport=1725,-2177,7532,3719&embedId=749907066196" frameborder="0" scrolling="no" allowfullscreen></iframe>

You can also access the miro board through the [link here](https://miro.com/app/board/uXjVOGRVlQc=/?invite_link_id=177508737098)

I also grouped some closely-related courses below together with their pre-requisite relations. Note that not all courses in the curriculum are included below. 

!!! note "general-purpose programming language"

    * INFO 1112 Principles of Program Structure and Design I (C++)
    * INFO 2313 Principles of Program Structure and Design II (Java)
    * INFO 2315 Data Structure (Java)
    * INFO 3245 Mobile Programming (Java)

    ``` mermaid
    graph LR
    title[<u>pre-requisite relation</u>]

    A[INFO 1112 ] --> B[INFO 2313];
    B[INFO 2313 ] --> C[INFO 3245];   
    A[INFO 2313 ] --> D[INFO 2315];   
    ```
!!! note "web development"


    * INFO 1213 Web Application Development (HTML, CSS, Javascript)
    * INFO 3135 Advanced Web Application Development (PHP, MySQL)
    * INFO 3225 Web Multimedia (Javascript)
    * INFO 4115 Human Factors and Website Design 
    * INFO 4235 Special Topics in Web and Mobile Application Development (modern Web frameworks)

    ``` mermaid
    graph LR
    title[<u>pre-requisite relation</u>]
    A[INFO 1213] --> B[INFO 3225]; 
    A[INFO 1213] --> C[INFO 3135];
    C[INFO 3135] --> D[INFO 4115];
    C[INFO 3135] --> E[INFO 4235];
    ```

!!! note "database"

    * INFO 2312 Database Management Systems
    * INFO 4330 Data Warehousing and Data Mining 

    ``` mermaid
    graph LR
    title[<u>pre-requisite relation</u>]
    A[INFO 2312] --> B[INFO 4330]; 
    ```

!!! note "computer security"

    * INFO 2411 Foundations of Computer Security
    * INFO 3171 System Security
    * INFO 4120 Digital Forensics
    * INFO 4125 Website and Cloud Security

    ```mermaid
    graph LR
    title[<u>pre-requisite relation</u>]
    A[INFO 2411] --> B[INFO 4125]; 
    A[INFO 2411] --> C[INFO 3171];
    C[INFO 3171] --> D[INFO 4120];
    ```

!!! note "cisco networking"


    * INFO 1212 Networking Technologies I
    * INFO 2311 Networking Technologies II
    * INFO 3390 Networking Technologies III
    * INFO 4260 Networking Technologies IV

    ```mermaid
    graph LR
    title[<u>pre-requisite relation</u>]
    A[INFO 1212] --> B[INFO 2311]; 
    B[INFO 2311] --> C[INFO 3390];
    C[INFO 3390] --> D[INFO 4260];
    ```

!!! note "wireless"
    * INFO 3180 Wireless Networks
    * INFO 4370 Security of Wireless Systems
    * INFO 4381 Internet of Things and Applications

    ```mermaid
    graph LR
    title[<u>pre-requisite relation</u>]
    A[INFO 3180] --> B[INFO 4381]; 
    A[INFO 3180] --> C[INFO 4370];
    ```

!!! note "system analysis and design"

    * INFO 1113 Systems Analysis and Design 
    * INFO 3150 Object-Oriented Software Engineering 

    No prerequisite relation exist between the two courses. 

!!! note "project"

    * INFO 2413 System Development Project
    * INFO 4190 Integration Project I
    * INFO 4290 Integration Project II

    ```mermaid
    graph LR
    title[<u>pre-requisite relation</u>]
    A[INFO 4190] --> B[INFO 4290]; 
    ```