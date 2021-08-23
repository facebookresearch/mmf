---
id: faqs
title: Frequently Asked Questions (FAQ)
sidebar_label: FAQs
---
#### 1. How to handle the segmentation fault? ####
It generally occurs when the program is trying to read or write an illegal memory location, for example, trying to read or write to a non-existent array element, not properly defined pointers etc.
It can be handled using the following utilities:-
* #### htop ####
  It allows us to monitor running processes on the system, their memory usage, CPU usage etc
  Here, looking into VIRT (Virtual memory usage), RES (Physical RAM usage in kb), SHR (Shared memory), PID (Process ID)   will give insights into the problem.
* #### vmstat ####
  Segfault's common reason is to access the part of the virtual address space that is not mapped to a physical one.
  vmstat (virtual memory statistics) might be helpful here, through this system activity can be observed in real-time.
  

