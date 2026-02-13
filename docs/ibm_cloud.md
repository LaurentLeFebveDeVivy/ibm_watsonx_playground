# IBM Cloud — Platform Overview for AI & Agent Workloads


This document focuses on the IBM Cloud service categories most relevant to building, deploying, and operating AI agent systems: **Compute**, **Storage**, **Networking & Security**, **Data & AI**, and **Integration**.

**Table of Contents**

1. [Compute](#1-compute)
2. [Storage](#2-storage)
3. [Networking & Security](#3-networking--security)
4. [Data & AI](#4-data--ai)
5. [Integration](#5-integration)
6. [How These Services Work Together for Agents](#6-how-these-services-work-together-for-agents)
- [References](#references)

---

## 1. Compute

Compute services provide the processing power for training models, running inference, hosting agent APIs, and executing background workloads. IBM Cloud offers a spectrum from fully managed serverless to dedicated bare-metal hardware.

### Virtual Servers (VPC)

Virtual servers on IBM Cloud VPC are multi-tenant virtual machines provisioned in minutes. They come in profiles optimized for different workloads: balanced (general-purpose), compute-optimized, and memory-optimized. For AI workloads, GPU-attached profiles support NVIDIA L40S, A100 PCIe, and Intel Gaudi 3 accelerators without needing dedicated hardware.

Security features include automatic self-encrypting physical disks (AES-256), support for customer-managed root keys (via Key Protect or Hyper Protect Crypto Services), and hardware-based confidential computing with Intel TDX and Intel SGX for workload isolation and integrity. Client-to-site VPN access is supported natively.

**Why it matters for agents:** Virtual servers are a good middle-ground for hosting agent application servers (e.g., a FastAPI backend) that need predictable performance but not dedicated hardware. They can scale vertically by changing profiles and horizontally via instance groups with auto-scaling. The confidential computing features (TDX, SGX) are particularly relevant for agents that process sensitive data — model inference can run inside a hardware-isolated enclave.

- [Virtual Servers for VPC — Product Page](https://www.ibm.com/products/virtual-servers)
- [Virtual Servers for VPC — Docs](https://cloud.ibm.com/docs/vpc?topic=vpc-about-advanced-virtual-servers)

### Bare-Metal Servers

Bare-metal servers are dedicated, single-tenant physical machines with no hypervisor overhead. IBM Cloud offers bare-metal servers with a range of AI accelerators:

| Accelerator | Use Case |
|---|---|
| NVIDIA H200 | Large-scale model training, high-throughput inference |
| NVIDIA L40S | Inference, fine-tuning, visualization |
| Intel Gaudi 3 | Cost-effective training and inference alternative |
| AMD MI300X | High-bandwidth memory workloads |

Configurations offer up to 11M+ combinations of CPU, memory, storage, and GPU, with 20 TB of free bandwidth included. Billing is available hourly, monthly, or on reserved 1- and 3-year terms for reduced rates.

For multi-GPU deployments, IBM Cloud provides advanced computing hubs (e.g., in Washington, D.C. and Frankfurt) with RoCE/RDMA networking enabling up to 3.2 Tbps GPU-to-GPU communication for distributed training.

**Why it matters for agents:** If you need to fine-tune foundation models on your own data, or run large models locally (instead of using watsonx.ai's hosted inference), bare-metal GPU servers give you the raw compute. This is the right choice for teams with heavy training or self-hosted inference requirements.

- [Bare Metal Servers — Overview](https://www.ibm.com/cloud/bare-metal-servers)
- [GPU Servers — NVIDIA on IBM Cloud](https://www.ibm.com/products/gpu-ai-accelerator/nvidia)

### IBM Kubernetes Service (IKS)

IKS is a managed Kubernetes service that handles the control plane, worker node provisioning, security patching, and version upgrades. It supports both VPC and classic infrastructure, multi-zone clusters for high availability, and integrates with IBM Cloud's IAM, logging, and monitoring services. For GPU workloads, IKS supports the GX3 flavor family with NVIDIA L4 Tensor Core GPUs (configurations from 1-GPU/16-core up to 4-GPU/64-core), enabling GPU-accelerated containers for inference and fine-tuning directly within Kubernetes pods.

**Why it matters for agents:** Agent systems typically consist of multiple cooperating services — an API gateway, the agent runtime, a vector database, a document indexer, monitoring dashboards. Kubernetes is the natural way to orchestrate these components, manage rolling deployments, and scale individual services independently. IKS removes the operational burden of managing Kubernetes itself.

**Problems it solves:**
- Orchestrating multi-container agent architectures
- Auto-scaling inference endpoints based on load
- Rolling deployments with zero downtime
- Service discovery between agent components

- [IBM Kubernetes Service — Docs](https://cloud.ibm.com/docs/containers?topic=containers-overview)
- [GX3 GPU Worker Nodes for IKS](https://www.ibm.com/new/announcements/introducing-gx3-kubernetes-and-openshift-clusters-with-nvidia-l4-gpus)

### Code Engine

Code Engine is a fully managed serverless platform that runs containers, batch jobs, and application code without any cluster management. You provide a container image or source code, and Code Engine handles provisioning, scaling (including scale-to-zero), and networking.


**Why it matters for agents:** Code Engine is ideal for deploying agent APIs that have variable traffic patterns. An agent endpoint that receives sporadic requests can scale to zero when idle (saving costs) and scale up automatically under load. It is also well-suited for batch processing tasks like document indexing — run a job to process a corpus, pay only for the execution time, and let it terminate. With GPU Fleets, you can even run local model inference serverlessly.

**Problems it solves:**
- Deploying agent APIs without managing infrastructure
- Running document ingestion or embedding jobs on a schedule
- Scale-to-zero for cost optimization on low-traffic services
- Quick iteration — push code, get an HTTPS endpoint

- [Code Engine — Docs](https://cloud.ibm.com/docs/codeengine?topic=codeengine-about)
- [Code Engine GPU Fleets](https://www.ibm.com/new/announcements/ibm-cloud-code-engine-introduces-serverless-fleets-with-gpus)

---

## 2. Storage

### IBM Cloud Object Storage (COS)

Cloud Object Storage is IBM's scalable, durable, and cost-effective storage service for **unstructured data**. It stores data as objects in buckets. COS supports S3-compatible APIs, making it interoperable with the broad ecosystem of tools built for Amazon S3.

Key features relevant to AI workloads:
- **Built-in Aspera high-speed transfer** — For moving large datasets (multi-GB/TB training corpora) quickly
- **SQL Query integration** — Run SQL directly on data stored in COS without needing to load it into a database
- **Immutable Object Storage** — WORM (Write Once Read Many) support for regulatory compliance
- **Integration with watsonx** — COS buckets can serve as data sources for watsonx.ai training jobs and watsonx.data lakehouse tables

**Why it matters for agents:** Every component of an agent system produces or consumes data. Document corpora for RAG need to be stored durably. Embedding models and fine-tuned model artifacts need a home. Conversation logs, evaluation datasets, and prompt templates all need persistent storage. COS provides a single layer for all of this, accessible via standard S3 APIs from Python (`ibm-cos-sdk` or `boto3`).

**Problems it solves:**
- Storing large document corpora for RAG indexing
- Persisting model artifacts and embeddings
- Archiving conversation logs and evaluation data
- Sharing datasets across team members and environments

- [Cloud Object Storage — Docs](https://cloud.ibm.com/docs/cloud-object-storage?topic=cloud-object-storage-about-cloud-object-storage)
- [ibm-cos-sdk Python Package](https://github.com/IBM/ibm-cos-sdk-python)

---

## 3. Networking & Security

### Virtual Private Cloud (VPC)

IBM Cloud VPC provides logically isolated network environments within the public cloud. Each VPC has its own address space, subnets, routing tables, access control lists (ACLs), and security groups. VPCs can span multiple availability zones within a region for high availability.

Connectivity options include:
- **IPsec site-to-site VPN** — Encrypted tunnels between your on-premises network and IBM Cloud VPC
- **Client-to-site VPN** — Remote developer access to VPC resources
- **Transit Gateway** — Connect multiple VPCs and classic infrastructure networks

**Why it matters for agents:** Agent systems often handle sensitive data — user queries, proprietary documents, API keys. VPC ensures that your compute resources (virtual servers, IKS clusters) run in an isolated network where you control all ingress and egress. The site-to-site VPN is particularly relevant when agents need to access on-premises data sources (e.g., enterprise databases behind a corporate firewall) for RAG retrieval.

- [VPC — Docs](https://cloud.ibm.com/docs/vpc?topic=vpc-about-vpc)
- [VPC Solutions](https://www.ibm.com/solutions/vpc)

### Identity and Access Management (IAM)

IBM Cloud IAM is the central service for managing authentication and authorization across all IBM Cloud resources. Core concepts:

- **Users and Service IDs** — Human users authenticate with IBMid; applications authenticate with Service IDs. Both receive API keys for programmatic access.
- **Access Groups** — Collections of users and service IDs that share the same access policies, simplifying permission management at scale.
- **Roles** — Two categories: *Platform management roles* (e.g., Viewer, Editor, Administrator) control account-level actions like provisioning instances. *Service access roles* (e.g., Reader, Writer, Manager) control service-specific actions like calling model inference APIs.

**Why it matters for agents:** Every call to watsonx.ai (and other IBM Cloud services) is authenticated via IAM API keys. Understanding IAM is essential for:
- Scoping permissions so the agent can only access the services it needs (principle of least privilege)
- Setting up service-to-service authorization (e.g., allowing Code Engine to pull from COS)
- Managing team access to shared watsonx.ai projects

- [IAM — Overview](https://cloud.ibm.com/docs/account?topic=account-iamoverview)

### Key Protect

Key Protect is a cloud-based key management service (KMS) for creating, managing, and rotating encryption keys. Keys are stored in FIPS 140-2 Level 3 certified hardware security modules (HSMs). Key Protect integrates with IBM Cloud services to provide envelope encryption — a pattern where a root key (stored in Key Protect) encrypts data encryption keys (DEKs) used by individual services.

Key features:
- **Bring Your Own Key (BYOK)** — Import your existing encryption keys into IBM Cloud's managed HSMs
- **Automatic key rotation** — Set rotation policies so keys are rotated on a schedule
- **Fine-grained access control** — Uses IAM policies to control who can create, rotate, or delete keys
- **Cross-region resiliency** — Keys replicated across regions with automatic failover

**Why it matters for agents:** Agent systems may process confidential enterprise data (documents, user queries, model outputs). Key Protect ensures that data at rest — whether in COS buckets, databases, or block storage — is encrypted with keys you control. Regulatory requirements (GDPR, HIPAA) often mandate customer-managed encryption keys.

- [Key Protect — Docs](https://cloud.ibm.com/docs/key-protect?topic=key-protect-about)

---

## 4. Data & AI

This is the core service category for agent development. The centerpiece is the **watsonx** suite, covered in depth in a [separate document](ibm_watsonx.md). Below is a brief orientation.

### The watsonx Suite

| Pillar | Purpose | Relevance to Agents |
|---|---|---|
| **watsonx.ai** | Foundation model development and deployment | The LLM backbone — prompt engineering, model inference, fine-tuning |
| **watsonx.data** | Data lakehouse for governed analytics | Feeds document stores and training data into agent pipelines |
| **watsonx.governance** | AI lifecycle governance and monitoring | Tracks agent behavior, detects bias and drift in production |

See [ibm_watsonx.md](ibm_watsonx.md) for detailed coverage of each pillar.

### Watson Discovery

Watson Discovery is a search and content intelligence service. It ingests documents (PDFs, web pages, structured data), builds search indexes, and provides natural language querying with passage retrieval. Discovery can serve as an enterprise-grade retrieval layer for RAG — an alternative to running your own vector database.

- [Watson Discovery — Docs](https://cloud.ibm.com/docs/discovery-data?topic=discovery-data-about)

### Watson Assistant

Watson Assistant is IBM's conversational AI service for building chatbots and virtual assistants. It provides dialog management, intent recognition, entity extraction, and integrations with messaging channels. While distinct from an agentic AI system, Watson Assistant can serve as a conversational front-end that routes complex queries to an agent backend.

- [Watson Assistant — Docs](https://cloud.ibm.com/docs/watson-assistant?topic=watson-assistant-about)

---

## 5. Integration

Integration services connect agent systems to the broader enterprise — databases, message queues, third-party APIs, and legacy systems.

### API Connect

API Connect is a full API lifecycle management platform: design APIs (OpenAPI/Swagger), enforce security policies (OAuth, rate limiting), proxy backend services, and publish API portals for consumers. It includes a developer portal for API documentation and key management.

**Why it matters for agents:** When you expose your agent as an API (e.g., a FastAPI endpoint), API Connect provides a production-grade gateway in front of it — handling authentication, rate limiting, request/response transformation, and analytics. This is how enterprise teams typically expose AI services to internal and external consumers.

- [API Connect — Docs](https://cloud.ibm.com/docs/apiconnect?topic=apiconnect-getting-started)

### Event Streams (Managed Kafka)

Event Streams is IBM's fully managed Apache Kafka service. It provides durable, high-throughput, real-time event streaming with schema registry support, built-in monitoring, and enterprise security. It exposes three APIs: Admin REST API (topic management), REST Producer API (HTTP-based message publishing), and Schema Registry API (Avro/JSON schema management). Hundreds of pre-built Kafka connectors are available for cloud, SaaS, and on-premises systems.

**Why it matters for agents:** Agents often need to react to events — a new document uploaded, a user action, a system alert. Event Streams provides the backbone for event-driven agent architectures where agents subscribe to event topics and act on incoming messages. It is also useful for streaming agent actions and observations to monitoring/logging systems. The REST Producer API is convenient for lightweight integrations where agents publish events via HTTP instead of the native Kafka protocol.

- [Event Streams — Docs](https://cloud.ibm.com/docs/EventStreams?topic=EventStreams-about)
- [Event Streams — Product Page](https://www.ibm.com/products/event-streams)

### App Connect

App Connect provides integration flows that connect cloud and on-premises applications without writing code. It includes pre-built connectors for hundreds of applications (Salesforce, SAP, Slack, databases) and supports event-driven or batch integration patterns.

**Why it matters for agents:** Agents that need to take actions in enterprise systems (update a CRM record, send an email, query an ERP) can use App Connect flows as tools. Instead of writing bespoke API integrations for each system, you configure connectors in App Connect and expose them as callable endpoints for the agent.

- [App Connect — Docs](https://cloud.ibm.com/docs/AppConnect?topic=AppConnect-about)

---

## 6. How These Services Work Together for Agents

A typical AI agent system on IBM Cloud combines these services:

```
                        ┌──────────────────────┐
                        │     API Connect      │  ← API gateway, rate limiting, auth
                        └──────────┬───────────┘
                                   │
                        ┌──────────▼──────────────┐
                        │  Code Engine / IKS      │  ← Agent runtime (FastAPI app)
                        │  ┌───────────────────┐  │
                        │  │  Agent (LangGraph)│  │
                        │  │  ┌─────┐ ┌─────┐  │  │
                        │  │  │Tools│ │ RAG │  │  │
                        │  │  └──┬──┘ └──┬──┘  │  │
                        │  └─────┼───────┼─────┘  │
                        └────────┼───────┼────────┘
                                 │       │
              ┌──────────────────┼───────┼──────────────────┐
              │                  │       │                  │
   ┌──────────▼────────┐  ┌──────▼───────▼──┐  ┌────────────▼───────────┐
   │  Enterprise APIs   │ │  watsonx.ai     │  │  Cloud Object Storage  │
   │  (via App Connect) │ │  (LLM inference)│  │  (documents, models)   │
   └────────────────────┘ └─────────────────┘  └────────────────────────┘
                                 │
                        ┌────────▼──────────┐
                        │ watsonx.governance│  ← Monitoring, bias detection
                        └───────────────────┘
```

The **security layer** (IAM, VPC, Key Protect) wraps across all services, ensuring authentication, network isolation, and data encryption.

---

## References

### IBM Cloud — General
- [IBM Cloud Documentation](https://cloud.ibm.com/docs)
- [IBM Cloud Catalog](https://cloud.ibm.com/catalog)
- [IBM Cloud Architecture Center](https://www.ibm.com/architectures)

### Compute
- [Virtual Servers for VPC](https://cloud.ibm.com/docs/vpc?topic=vpc-about-advanced-virtual-servers)
- [Bare Metal Servers](https://www.ibm.com/cloud/bare-metal-servers)
- [NVIDIA GPUs on IBM Cloud](https://www.ibm.com/products/gpu-ai-accelerator/nvidia)
- [IBM Kubernetes Service](https://cloud.ibm.com/docs/containers?topic=containers-overview)
- [Code Engine](https://cloud.ibm.com/docs/codeengine?topic=codeengine-about)

### Storage
- [Cloud Object Storage](https://cloud.ibm.com/docs/cloud-object-storage?topic=cloud-object-storage-about-cloud-object-storage)

### Security
- [IAM Overview](https://cloud.ibm.com/docs/account?topic=account-iamoverview)
- [Key Protect](https://cloud.ibm.com/docs/key-protect?topic=key-protect-about)
- [Hyper Protect Crypto Services](https://cloud.ibm.com/docs/hs-crypto?topic=hs-crypto-overview)

### Data & AI
- [watsonx — Product Page](https://www.ibm.com/watsonx)
- [Watson Discovery](https://cloud.ibm.com/docs/discovery-data?topic=discovery-data-about)
- [Watson Assistant](https://cloud.ibm.com/docs/watson-assistant?topic=watson-assistant-about)

### Integration
- [API Connect](https://cloud.ibm.com/docs/apiconnect?topic=apiconnect-getting-started)
- [Event Streams](https://cloud.ibm.com/docs/EventStreams?topic=EventStreams-about)
- [App Connect](https://cloud.ibm.com/docs/AppConnect?topic=AppConnect-about)
