# Lightweight-Anomaly-Detection-for-Network-Flows-Capstone-Project

## Motivation
Network security is really important because of the growing number of cyber threats. Traditional intrusion detection systems still struggle to identify zero day threats. This has resulted in a lot of research in anomaly based detection methods. These methods can detect deviations from normal network behaviour without any previous knowledge. But a lot of these systems are dependent on very complex architectures that require a lot of computational resources. This makes them almost impractical for many environments like IOTs, edge devices and real time monitoring systems. This project focuses on the gap between detection and computational effectiveness by making a lightweight autoencoder that is based on anomaly detection systems for network flows.

## Research Questions
Detection Effectiveness: Can a simple autoencoder that is trained only on normal traffic still compete with more complex methods  when tested?
Threshold Selection: How do you decide where to draw the line between normal and suspicious behaviour once the model is trained?
Attack Generalisation: Can the model still pick up on different types of attacks even though it has only ever seen normal traffic?
Efficiency vs Performance: How much detection capability can you realistically get out of a lightweight model?
Adversarial Attacks: How easily can this autoencoder be fooled by attempts to slip past it and could training it on slightly noisy data offer any protection?

## Methodology
Three fully connected autoencoders of increasing size were trained using normal traffic with no attack labels involved. The models are small at around 30K parameters and 117 KB, medium at around 70K parameters and 273 KB, and large at around 180K parameters and 706 KB. Anomaly detection was based on reconstruction error, and three different threshold strategies were tested. Percentile based, standard deviation based, and ROC optimal. 

## Key Findings
Detection Performance: The medium autoencoder came out on top with a ROC-AUC of 0.8853 and a PR-AUC of 0.9477. The large model had 2.6 times more parameters but did worse. 
Threshold Selection: The 99th percentile of validation reconstruction error was the most practical choice. It needs no labelled data and it catches 74.1% of true attacks but keeps the false positive rate down to just 3.6%.
Attack Generalisation: High volume attacks with clear structure were detected like DoS at over 93%, and Backdoor at over 92%. But  attacks designed to blend in with normal traffic like Shellcode detection were as low as 4.9% on the small model and only 27.5% on the medium.
Efficiency vs Performance: All models ran with inference times under 0.002 ms per sample. Going from small to medium was actually worth it gains without much of a size penalty.
Adversarial Robustness: The small model was the most vulnerable and dropped from 68.1% detection down to 30.4% at epsilon 0.10 under FGSM attacks. The denoising defence cuts detection by 31% under attack.

## Conclusion And Contributions
This work makes a reproducible baseline on UNSW-NB15, a comparison of efficiency and performance for different architectures, threshold guidance that does not need labels, and an adversarial robustness result that alsohas a clear proof against using denoising as a defence.
Effective network intrusion detection does not require large or expensive models. A medium autoencoder with fewer than 70,000 parameters and trained only on normal traffic can achieve strong real time detection and is small enough to run on edge devices. That makes it a realistic option for organisations that cant afford complex security. It would also be goog to explore hybrid supervised and unsupervised architectures to catch stealthy attacks and testing these models on live network traffic.

