---
title: "TopoNet — Graph-based Topology Reasoning for Driving Scenes"
subtitle: "Paper review · the first end-to-end framework and dataset that reason about how lanes connect — to each other, and to traffic elements."
date: 2026-06-04 19:00:00 +0900
categories: [Lane Topology Reasoning]
tags: [lane-topology-reasoning, autonomous-driving, graph-neural-network]
math: true
toc: true
bilingual: true
---

<div class="lang-block" lang="ko" markdown="1">

각 figure의 출처는 하이퍼링크로 달아두었습니다 :)

Online Mapping을 공부하다 보면, Centerline만 Detection을 하는 것에 대해 의문이 생깁니다. Planning 입장에서

*+++ Lane Topology Reasoning 섹션의 첫 글이다. Online HD Map Construction 쪽 논문들을 한참 읽다 보니, 결국 perception이 잘 되는 것만으로는 부족하고 "그래서 이 차선에서 저 차선으로 갈 수 있느냐"라는 관계(topology)를 풀어야 진짜 주행에 쓸 수 있는 지도가 된다는 걸 느꼈다. 그 출발점이 바로 이 TopoNet이라 생각해서 첫 글로 골랐다 :rocket:*

<br><br>




> **TopoNet: Graph-based Topology Reasoning for Driving Scenes**

> > **Authors : Tianyu Li, Li Chen, Huijie Wang, Yang Li, Jiazhi Yang, Xiangwei Geng, Shengyin Jiang, Yuting Wang, Hang Xu, Chunjing Xu, Junchi Yan, Ping Luo, Hongyang Li (OpenDriveLab)**

> > **Paper Link : https://arxiv.org/abs/2304.05277**

> > **Code : https://github.com/OpenDriveLab/TopoNet**



<br><br>



## **Introduction & Motivation**
---
### **왜 Topology인가?**
---

자율주행 스택에서 지도(map)의 역할은 단순히 "차선이 여기 있다"를 아는 데서 끝나지 않는다. 진짜로 필요한 건 **"지금 이 차선에서 어디로 진입할 수 있는가"**, 그리고 **"저 신호등/표지판이 어느 차선을 통제하는가"** 같은 *관계*다.

기존의 Online HD Map Construction 계열(HDMapNet, MapTR, VectorMapNet 등)은 차선, 경계선, 횡단보도 같은 map element를 **검출(detection)**하는 데 집중했다. 하지만 이렇게 검출된 element들 사이의 **연결 관계(topology)**는 명시적으로 다루지 않거나, 후처리(post-processing)로 휴리스틱하게 붙이는 경우가 많았다.

TopoNet은 여기서 한 걸음 더 나아가, 다음 네 가지를 **하나의 end-to-end 프레임워크 안에서 동시에** 푼다.

1. **Lane Centerline Detection** — 차선의 중심선(directed)을 검출
2. **Traffic Element Detection** — 신호등, 표지판 등 교통 요소를 검출
3. **Lane-Lane Topology** — 차선과 차선이 어떻게 이어지는지 (진입/분기/합류)
4. **Lane-Traffic Topology** — 어떤 교통 요소가 어떤 차선을 통제하는지

저자들은 이를 두고 *"abstracting traffic knowledge beyond conventional perception tasks"* 라고 표현한다. 즉, **검출(perception)을 넘어선 추론(reasoning)**이 핵심이다.

<div style="text-align: center;">
     <a href = "https://arxiv.org/abs/2304.05277" style = "color: inherit; text-decoration: none;">
         <b>Fig 1. TopoNet은 lane–lane / lane–traffic 관계를 그래프로 추론한다 (논문 Fig. 1 참고)</b></a> <br>
</div>

---

### **왜 Centerline인가? (vs. Lane Divider)**

여기서 짚고 넘어갈 포인트가 하나 있다. 기존 mapping 논문들이 주로 다루던 것은 **lane divider(차선 경계선)**였다. 반면 topology를 다루려면 **lane centerline(차선 중심선)**이 더 적합하다.

이유는 간단하다. 경계선은 "공간을 나누는 선"일 뿐, 방향성이나 연결성을 직접 담지 못한다. 반면 centerline은 **방향이 있는(directed) 경로**로 볼 수 있어서, 한 centerline의 끝점이 다른 centerline의 시작점과 이어진다는 식으로 **그래프의 간선(edge)**을 자연스럽게 정의할 수 있다.

즉, centerline을 노드로, 연결 관계를 간선으로 보면 주행 가능한 경로 전체가 하나의 **방향 그래프(directed graph)**가 된다. 이것이 topology reasoning의 출발점이다.

---

## **Method**
---
### **전체 파이프라인 개요**

TopoNet의 큰 그림은 다음과 같다.

- **Feature Extraction** — multi-view 카메라 이미지에서 BEV(Bird's-Eye-View) feature와 PV(Perspective-View) feature를 뽑는다. Lane은 BEV 공간에서, traffic element는 원근 영상(PV) 공간에서 검출하는 것이 자연스럽다.
- **Query 기반 Detection** — DETR 계열처럼 lane query와 traffic query를 두고, 각각 centerline과 traffic element를 디코딩한다.
- **Scene Graph Neural Network (SGNN)** — 검출된 query들을 노드로 삼아, 그들 사이의 관계를 그래프 위에서 메시지 패싱으로 갱신한다. **이 부분이 TopoNet의 핵심 기여다.**
- **Topology Head** — 갱신된 feature로부터 lane-lane, lane-traffic 연결 여부를 예측한다.

---

### **Scene Graph Neural Network (SGNN)**

핵심은 SGNN이다. 직관은 이렇다 — *"각 차선의 feature를, 그 차선과 이어진 이웃 차선과 그 차선을 통제하는 교통 요소의 정보로 보강하자."*

기존 방식의 한계는, 각 element를 **독립적으로** 디코딩한 뒤 마지막에 관계만 따로 예측한다는 점이었다. 그러면 element feature 자체에는 "내가 누구와 이어져 있는가"라는 맥락이 들어있지 않다.

SGNN은 이를 뒤집는다. **관계 정보를 feature를 만드는 과정 안으로 끌어들인다.** 차선 노드 $i$의 feature를 업데이트할 때, 그와 연결될 후보인 이웃 노드들의 feature를 모아(aggregate) 반영한다. 대략적으로 쓰면 메시지 패싱은

$$
\mathbf{h}_i^{(l+1)} = \phi\Big( \mathbf{h}_i^{(l)},\ \underset{j \in \mathcal{N}(i)}{\text{Agg}}\ \psi\big(\mathbf{h}_i^{(l)}, \mathbf{h}_j^{(l)}\big) \Big)
$$

형태가 된다. 여기서 $\mathcal{N}(i)$는 노드 $i$의 이웃(연결 후보)들이고, $\psi$는 두 노드 사이의 관계 메시지를, $\phi$는 자기 자신의 feature와 모아진 메시지를 결합하는 함수다.

이렇게 graph 위에서 feature를 갱신하면, **검출(detection)과 추론(reasoning)이 서로를 강화한다.** 연결 관계를 알면 차선 검출이 좋아지고, 차선이 잘 검출되면 연결 관계 추론이 좋아진다. 이 선순환이 SGNN이 노리는 지점이다.

<div style="text-align: center;">
     <a href = "https://github.com/OpenDriveLab/TopoNet" style = "color: inherit; text-decoration: none;">
         <b>Fig 2. SGNN을 통한 scene graph 기반 feature 갱신 (논문/레포 architecture 참고)</b></a> <br>
</div>

---

### **Lane–Lane vs. Lane–Traffic 두 종류의 관계**

TopoNet이 푸는 topology는 두 가지인데, 성격이 꽤 다르다.

- **Lane–Lane (TOP$_{ll}$)** — 같은 BEV 공간 안에서, directed centerline들 사이의 연결을 본다. 끝점-시작점이 이어지면 간선이 생긴다. 둘 다 같은 modality(BEV)라 비교가 비교적 직접적이다.
- **Lane–Traffic (TOP$_{lt}$)** — 한쪽은 BEV의 차선, 다른 쪽은 PV(원근 영상)의 신호등/표지판이다. **서로 다른 좌표계·modality를 잇는 cross-view 관계**라서 더 까다롭다. "이 신호등이 이 차선을 통제한다"를 맞히려면 두 공간의 feature를 한 그래프 위에서 정렬해야 한다.

SGNN은 이 두 종류의 관계를 모두 같은 graph 메커니즘 안에서 처리한다.

---

## **OpenLane-V2 Benchmark & OLS Metric**
---

TopoNet은 단순히 모델만 내놓은 게 아니라, 이 task를 평가하기 위한 벤치마크 **OpenLane-V2**와 함께 등장했다 (벤치마크는 NeurIPS 2023 Datasets & Benchmarks 트랙에 별도로 발표됨).

평가지표 **OLS(OpenLane-V2 Score)**는 네 가지 하위 지표의 종합이다.

- **DET$_l$** — lane centerline 검출 정확도
- **DET$_t$** — traffic element 검출 정확도
- **TOP$_{ll}$** — lane-lane topology 정확도
- **TOP$_{lt}$** — lane-traffic topology 정확도

$$
\text{OLS} = \tfrac{1}{4}\Big( \text{DET}_{l} + \text{DET}_{t} + \sqrt{\text{TOP}_{ll}} + \sqrt{\text{TOP}_{lt}} \Big)
$$

topology 항에 제곱근이 붙는 이유는, topology 점수가 detection 점수보다 일반적으로 훨씬 낮게 나오기 때문이다(관계 추론이 검출보다 어렵다). 제곱근으로 스케일을 끌어올려 네 지표가 비슷한 영향력을 갖도록 균형을 맞춘 것이다.

### **결과**

OpenLane-V2 subset-A validation 기준(v1.1 metric), TopoNet의 대략적인 수치는 다음과 같다.

| 지표 | 값 |
|---|---|
| **OLS** | **39.8** |
| DET$_l$ (centerline) | 28.6 |
| DET$_t$ (traffic element) | 48.6 |
| TOP$_{ll}$ (lane-lane) | 10.9 |
| TOP$_{lt}$ (lane-traffic) | 23.8 |

핵심은 절대 수치 자체보다, **topology 지표(TOP$_{ll}$, TOP$_{lt}$)에서 기존 mapping 계열(MapTR, VectorMapNet 등)을 큰 폭으로 앞섰다는 점**이다. detection만 잘하던 모델들과 달리, 관계 추론을 명시적으로 설계에 넣은 효과가 여기서 드러난다.

다만 TOP$_{ll}$이 10.9에 불과하다는 점은, **lane-lane topology가 여전히 매우 어려운 미해결 문제**라는 것도 동시에 보여준다. 이 지점이 이후 후속 연구들이 파고드는 틈이 된다.

---

## **Conclusion & 이 섹션에서의 위치**
---

TopoNet의 기여를 한 줄로 요약하면 —

> **차선·교통요소 검출과 그들 사이의 관계 추론을, 별개의 단계가 아니라 하나의 graph 위에서 end-to-end로 묶은 첫 프레임워크.**

기존 Online Mapping이 "무엇이 어디 있는가"를 풀었다면, TopoNet은 거기에 "**그래서 그것들이 어떻게 이어지는가**"를 더한다. 그리고 그 관계야말로, 검출된 지도를 실제 주행 가능한 경로 그래프로 바꿔주는 마지막 조각이다.

이 글을 Lane Topology Reasoning 섹션의 첫 글로 둔 이유가 여기에 있다. 이후에 다룰 **TopoMLP**(graph 대신 강한 detection + 단순한 MLP topology head로 더 좋은 성능을 낸다는 반박), **LaneSegNet**(lane segment를 map learning의 단위로 삼는 표현) 같은 논문들은 모두 TopoNet이 정의한 문제 설정 위에서 출발한다.

다음 글에서는 그중 TopoMLP를 다뤄보려 한다.

---

> 읽어주셔서 감사합니다. 혹시 제가 잘못 이해한 부분이 있다면 언제든 알려주세요 :)

</div>

<div class="lang-block" lang="en" markdown="1">

Sources for each figure are linked inline :)

Once you spend enough time studying Online Mapping, a question creeps in: is it really enough to *only* detect centerlines? From a planning standpoint —

*+++ This is the first post in the Lane Topology Reasoning section. After reading through a stack of Online HD Map Construction papers, I came to feel that good perception alone isn't enough — you have to solve the *relationships* (topology), like "can I actually go from this lane to that one," before a map becomes something you can really drive on. TopoNet felt like the natural starting point for that, so I picked it as the first post :rocket:*

<br><br>




> **TopoNet: Graph-based Topology Reasoning for Driving Scenes**

> > **Authors : Tianyu Li, Li Chen, Huijie Wang, Yang Li, Jiazhi Yang, Xiangwei Geng, Shengyin Jiang, Yuting Wang, Hang Xu, Chunjing Xu, Junchi Yan, Ping Luo, Hongyang Li (OpenDriveLab)**

> > **Paper Link : https://arxiv.org/abs/2304.05277**

> > **Code : https://github.com/OpenDriveLab/TopoNet**



<br><br>



## **Introduction & Motivation**
---
### **Why topology?**
---

In an autonomous-driving stack, a map's job doesn't end at knowing "a lane is here." What you actually need are the *relationships*: **"from this lane, where can I go next?"** and **"which lane does that traffic light / sign govern?"**

Prior Online HD Map Construction methods (HDMapNet, MapTR, VectorMapNet, etc.) focused on **detecting** map elements — lanes, dividers, crosswalks. But the **connectivity (topology)** between those detected elements was either left implicit or stitched on afterwards with hand-crafted post-processing heuristics.

TopoNet goes a step further, solving all four of the following **jointly, inside a single end-to-end framework**:

1. **Lane Centerline Detection** — detect directed lane centerlines
2. **Traffic Element Detection** — detect traffic lights, signs, and other elements
3. **Lane-Lane Topology** — how lanes connect to each other (merge / split / continue)
4. **Lane-Traffic Topology** — which traffic element governs which lane

The authors describe this as *"abstracting traffic knowledge beyond conventional perception tasks."* In other words, the key is **reasoning that goes beyond perception**.

<div style="text-align: center;">
     <a href = "https://arxiv.org/abs/2304.05277" style = "color: inherit; text-decoration: none;">
         <b>Fig 1. TopoNet reasons about lane–lane / lane–traffic relations as a graph (see Fig. 1 of the paper)</b></a> <br>
</div>

---

### **Why centerlines? (vs. lane dividers)**

One point worth pausing on: earlier mapping papers mostly dealt with **lane dividers** (the boundary lines). For topology, however, **lane centerlines** are the better primitive.

The reason is simple. A divider is just "a line that splits space" — it doesn't directly carry direction or connectivity. A centerline, on the other hand, can be seen as a **directed path**, so you can naturally define a **graph edge** by saying "the endpoint of one centerline connects to the start of another."

Treat centerlines as nodes and their connections as edges, and the entire set of drivable routes becomes a single **directed graph**. That is the starting point of topology reasoning.

---

## **Method**
---
### **Pipeline overview**

The big picture of TopoNet looks like this:

- **Feature Extraction** — from multi-view camera images, extract both BEV (Bird's-Eye-View) and PV (Perspective-View) features. Lanes are naturally detected in BEV space, traffic elements in the perspective (PV) image.
- **Query-based Detection** — following the DETR family, use lane queries and traffic queries to decode centerlines and traffic elements respectively.
- **Scene Graph Neural Network (SGNN)** — treat the detected queries as nodes and update their features through message passing over a graph. **This is TopoNet's core contribution.**
- **Topology Head** — from the updated features, predict lane-lane and lane-traffic connectivity.

---

### **Scene Graph Neural Network (SGNN)**

SGNN is the heart of the method. The intuition: *"enrich each lane's feature with information from its connected neighbor lanes and from the traffic elements that govern it."*

The limitation of prior approaches was that each element was decoded **independently**, with the relationships predicted only at the very end. That leaves the element features themselves without any context about "who am I connected to."

SGNN flips this around. **It pulls relational information into the feature-building process itself.** When updating the feature of lane node $i$, it aggregates the features of its candidate neighbor nodes. Roughly, the message passing takes the form

$$
\mathbf{h}_i^{(l+1)} = \phi\Big( \mathbf{h}_i^{(l)},\ \underset{j \in \mathcal{N}(i)}{\text{Agg}}\ \psi\big(\mathbf{h}_i^{(l)}, \mathbf{h}_j^{(l)}\big) \Big)
$$

where $\mathcal{N}(i)$ is the set of neighbors (connection candidates) of node $i$, $\psi$ produces the relational message between two nodes, and $\phi$ combines a node's own feature with the aggregated messages.

By updating features over the graph this way, **detection and reasoning reinforce each other.** Knowing the connections improves lane detection, and well-detected lanes improve connectivity reasoning. That virtuous cycle is exactly what SGNN is after.

<div style="text-align: center;">
     <a href = "https://github.com/OpenDriveLab/TopoNet" style = "color: inherit; text-decoration: none;">
         <b>Fig 2. Scene-graph-based feature update via SGNN (see the paper / repo architecture)</b></a> <br>
</div>

---

### **Lane–Lane vs. Lane–Traffic: two kinds of relations**

TopoNet solves two kinds of topology, and they have quite different characters.

- **Lane–Lane (TOP$_{ll}$)** — within the same BEV space, this looks at connections between directed centerlines. An edge forms when one's endpoint meets another's start. Since both live in the same modality (BEV), the comparison is relatively direct.
- **Lane–Traffic (TOP$_{lt}$)** — here one side is a lane in BEV and the other is a traffic light / sign in PV (the perspective image). This is a **cross-view relation linking different coordinate frames and modalities**, which makes it trickier. To get "this light governs this lane" right, you have to align features from the two spaces on a single graph.

SGNN handles both kinds of relations within the same graph mechanism.

---

## **OpenLane-V2 Benchmark & OLS Metric**
---

TopoNet didn't just ship a model — it arrived together with **OpenLane-V2**, a benchmark for evaluating this task (the benchmark itself was published separately in the NeurIPS 2023 Datasets & Benchmarks track).

The evaluation metric, **OLS (OpenLane-V2 Score)**, is a composite of four sub-metrics:

- **DET$_l$** — lane centerline detection accuracy
- **DET$_t$** — traffic element detection accuracy
- **TOP$_{ll}$** — lane-lane topology accuracy
- **TOP$_{lt}$** — lane-traffic topology accuracy

$$
\text{OLS} = \tfrac{1}{4}\Big( \text{DET}_{l} + \text{DET}_{t} + \sqrt{\text{TOP}_{ll}} + \sqrt{\text{TOP}_{lt}} \Big)
$$

The square root on the topology terms is there because topology scores tend to come out much lower than detection scores (relational reasoning is harder than detection). Taking the square root lifts their scale so that all four sub-metrics carry comparable weight.

### **Results**

On OpenLane-V2 subset-A validation (v1.1 metric), TopoNet's approximate numbers are:

| Metric | Value |
|---|---|
| **OLS** | **39.8** |
| DET$_l$ (centerline) | 28.6 |
| DET$_t$ (traffic element) | 48.6 |
| TOP$_{ll}$ (lane-lane) | 10.9 |
| TOP$_{lt}$ (lane-traffic) | 23.8 |

What matters is not the absolute numbers but the fact that **on the topology metrics (TOP$_{ll}$, TOP$_{lt}$) it beats the prior mapping line (MapTR, VectorMapNet, etc.) by a wide margin.** Unlike models that were only good at detection, the effect of explicitly building relational reasoning into the design shows up right here.

That said, a TOP$_{ll}$ of just 10.9 also shows that **lane-lane topology remains a very hard open problem.** That gap is exactly the opening that later work digs into.

---

## **Conclusion & where this sits in the section**
---

TopoNet's contribution in one line —

> **The first framework to bind lane/traffic-element detection and the reasoning about their relationships into a single graph, end-to-end, rather than as separate stages.**

Where earlier Online Mapping solved "what is where," TopoNet adds "**and so, how do those things connect.**" And that connectivity is precisely the final piece that turns a detected map into an actually drivable route graph.

That is why this post opens the Lane Topology Reasoning section. The papers I'll cover later — **TopoMLP** (a rebuttal arguing that strong detection plus a simple MLP topology head does even better than elaborate graph reasoning) and **LaneSegNet** (a representation that takes the lane segment as the unit of map learning) — all set out from the problem formulation TopoNet defined.

In the next post, I'll dig into TopoMLP.

---

> Thanks for reading. If I've misunderstood anything, please don't hesitate to let me know :)

</div>
