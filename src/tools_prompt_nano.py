# -*- coding: utf-8 -*-
"""
Filename: tools_prompt_nano.py
Description: Local prompt bundle for the packaged GRID repository. This module vendors the prompt functions required by train-data, src/grid, src/baseline, and eval so the artifact does not depend on the Dropbox-level tools_prompt.py.
Keywords: GRID, prompt bundle, train-data, evaluation, judge prompt
"""

from __future__ import annotations

import json


grid_entity_type_definition = '''
###Main Topic:Actors and Identities###
Name:'user-account'
Definition:Represents a user account that can be used to log in and access system resources.
Note:This object is used to describe account information involved in an attack. Its key properties include the user ID (e.g., SID for Windows or UID for Linux), login name, and display name. This object can represent both human user accounts (e.g., 'user123') and system or service accounts (e.g., 'SYSTEM' or 'root'), which is crucial for analyzing attack behaviors like privilege escalation and lateral movement.
Example:NT AUTHORITY\SYSTEM

Name:'identity'
Definition:An entity that represents a person, organization, or group.
Note:This is a broad category for any formally identified entity. It must be distinguished from 'threat-actor-or-intrusion-set'. An 'identity' can be a victim, a reporting party, a researcher, or a benign company. Only when an 'identity' is determined to be malicious should it be classified as a 'threat-actor-or-intrusion-set'.
Example:Google Threat Analysis Group, FireEye

Name:'threat-actor-or-intrusion-set'
Definition:A combined entity that can represent an individual, group, or organization conducting cyberattacks (i.e., a 'Threat Actor'), or a set of attack activities with common goals, tactics, and infrastructure (i.e., an 'Intrusion Set').
Note:This type merges the attacker itself with its cluster of activity. When a threat actor has a clear name (e.g., APT28), use its name directly. If there is no clear name, use it original text in the context to call it, such as 'the attacker' or 'the group that using XYZ malware' (if it is appeared in the text). 
Example:APT28, Sandworm Team.

###Main Topic:Malicious Code and Tools###
Name:'detailed-part-of-malware-or-hackertool'
Definition:A specific component, function, module, or configuration block that is an integral part of a 'malware' or 'hacker-tool'.
Note:This describes the internal, often custom-built, workings of malicious software. It is the component-level counterpart to 'malware' and 'hacker-tool'. For components of legitimate software, use 'detailed-part-of-general-software'.
Example:The EternalBlue exploit module in WannaCry; a specific function named keylog_routine().

Name:'malware'
Definition:Software designed to be executed on a victim's system to cause harm, steal data, or establish unauthorized control. It functions as the malicious payload or "ammunition" in an attack.
Note:This entity represents the part of the attack that runs within the victim's environment, often automatically or silently. While some systems (like RATs) have components of both, 'malware' specifically refers to the client-side agent. Examples include ransomware (WannaCry), trojans (Emotet), and spyware (Agent Tesla).
Example: WannaCry, Emotet, Agent Tesla

Name:'hacker-tool'
Definition:Software used by an attacker or security professional to orchestrate, control, or facilitate a cyberattack. It functions as the "workbench" or "control panel" for the operation.
Note:This entity represents the part of the attack that typically runs within the attacker's environment. It is often interactive and used to create payloads, manage infrastructure, or control malware on victim systems. For the server/control component of a RAT, this type should be used. Examples include exploitation frameworks (Metasploit), C2 platforms (Cobalt Strike), and network scanners (Nmap). Only the name should be used.
Example: Cobalt Strike, Metasploit Framework, Mimikatz.

###Main Topic:Legitimate Software###
Name:'detailed-part-of-general-software'
Definition:A specific component, function, library, or command that is part of a 'general-software' entity and is being leveraged in a malicious context.
Note:This allows for a granular description of how legitimate software is abused. It is the component-level counterpart to 'general-software'.
Example:The Invoke-Expression cmdlet in PowerShell; the CreateRemoteThread Windows API function.

Name:'general-software'
Definition:Legitimate software with a non-malicious primary purpose that is either exploited as a target of an attack (e.g., via a vulnerability), serves as a host for malicious code, or is abused by threat actors to facilitate an attack.
Note:This is the counterpart to 'hacker-tool' and is central to Living off the Land (LotL) attacks. For its specific components, use 'detailed-part-of-general-software'.
Example: Microsoft Office, PowerShell, curl, OpenSSH

###Main Topic:Attack Actions and Campaigns###
Name:'vulnerability'
Definition:A flaw or weakness in software, hardware, or procedure that a threat actor can exploit to cause harm.
Note:This entity represents the weakness itself, not the act of exploiting it. An 'attack-pattern' is what 'exploits' a 'vulnerability'. It is most often identified by a formal tracking number like a CVE identifier.
Example:CVE-2021-44228

Name:'attack-pattern'
Definition:A specific attack method or technique described verbatim in the source text that an adversary uses to achieve a malicious objective.
Note:Indicates the exact attack method or technique mentioned in the source text, without mapping to any predefined TTP framework identifiers; retains the terminology used by the original author.
Example:Phishing email delivery, SQL injection attempt, Brute-force login operation

Name:'campaign'
Definition:Refers to a series of attack actions with specific objectives and a timeframe, typically launched by one or more threat actors to achieve a strategic goal.
Note:A campaign usually has a well-known name (e.g., 'Operation Shady RAT'). If not name is given, use the original text in the context to call it, such as 'the attack' (which is appeared in the text).
Example: Operation Shady RAT

###Main Topic:Host-based Observables###
Name:'windows-registry-key'
Definition:Represents a key in the Windows Registry, which is a hierarchical database used for storing settings for the operating system and applications.
Note:The registry key is critical for analyzing malicious activity on Windows systems. Malware often achieves persistence, stores configuration, or disables security software by modifying or creating registry keys. The key properties of this object include the hive, the key path, and one or more values contained within the key, each of which has a name, data, and type.
Example:HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Run

Name:'process'
Definition:An instance of a computer program that is being executed. It contains all the state information of the program at runtime.
Note:This is a core object for host-based analysis. Its key properties include the Process ID (PID), the process name (typically the executable file path), its command-line arguments, and a reference to its parent process (parent_ref). By analyzing processes and their parent-child relationships, a clear execution chain can be constructed, such as a Word document process launching a PowerShell process.
Example:powershell.exe:3104

Name:'file'
Definition:A computer file, which is a resource for storing information in a computer system.
Note:This entity should be distinguished from 'malware'. A 'file' is the technical object (a sequence of bytes with a name and path), while 'malware' is a classification of that file's malicious purpose. Any 'malware' entity is represented by a 'file' object, but not all 'file' objects are malware (e.g., a benign document targeted by an exploit). It is also different from a 'process', which is the running instance of a file.
Example:C:\Windows\System32\ntdll.dll

###Main Topic:Network Observables and Infrastructure###
Name:'url'
Definition:A Uniform Resource Locator, providing a complete, specific address for a resource on the World Wide Web.
Note:The key distinction is with 'domain-name'. A URL is more specific, containing the scheme (e.g., http, ftp), the domain, and often a port, path, and query string. Many different URLs can exist on a single domain.
Example:http://phishing-site.com/login.php?user=victim

Name:'domain-name'
Definition:A human-readable name that corresponds to a network resource, typically an IP address, as part of the Domain Name System (DNS).
Note:This entity should be distinguished from a 'url'. A 'domain-name' is only the host part (e.g., example.com), while a 'url' is the full address including the protocol and path (e.g., https://example.com/login). It is also distinct from the IP address(es) it 'resolves-to'.
Example:evil-c2-server.net

Name:'ipv4-addr'
Definition:An Internet Protocol version 4 (IPv4) address, which is a numerical label assigned to each device participating in a computer network that uses the Internet Protocol for communication.
Note:This is a fundamental network observable. It must be distinguished from a 'mac-address' (Layer 3 logical address vs. Layer 2 physical address) and a 'domain-name' (machine-routable address vs. human-readable name). An IP address by itself is just an observable; it only becomes an 'indicator' when context about its maliciousness is added.
Example:198.51.100.10

Name:'ipv6-addr'
Definition:An Internet Protocol version 6 (IPv6) address, representing a logical network address for modern internet protocols.
Note:Functionally similar to 'ipv4-addr' but with a much larger address space. All distinctions that apply to 'ipv4-addr' (e.g., vs. 'mac-address' or 'domain-name') also apply here.
Example:2001:0db8:85a3:0000:0000:8a2e:0370:7334

Name:'network-traffic'
Definition:Represents an aggregation of network traffic that flows between two or more network endpoints. It can be a single connection or a series of related network packets.
Note:This object does not represent a single packet but describes a "flow". It uses references to other objects to fully define this flow, such as source and destination IP addresses (src_ref, dst_ref), source and destination ports (src_port, dst_port), and the protocols used (protocols). It is often used to describe network-level activities like C2 communications, data exfiltration, or network scanning. In a graph, a traffic node is often named using a 5-tuple.
Example:tcp:192.168.1.100:51234-203.0.113.10:443

Name:'mac-address'
Definition:Refers to the Media Access Control Address, a unique physical address assigned to a network interface controller (NIC) for communication at the data link layer.
Note:A MAC address is typically a 48-bit address, represented as six groups of hexadecimal digits separated by colons or hyphens. It operates at Layer 2 (Data Link Layer) of the OSI model, which is different from an IP address that operates at Layer 3 (Network Layer). An IP address is a logical address that can be changed, while a MAC address is a physical address burned into the hardware. It is used to deliver data frames to the correct device on a local network.
Example:00:1A:2B:3C:4D:5E

Name:'email-address'
Definition:A specific address used to send and receive electronic mail messages.
Note:This is a fundamental observable object. It is a key component in phishing 'attack-patterns' and can be associated with an 'identity' or used as a login for a 'user-account'.
Example:phisher@example.com

Name:'infrastructure'
Definition:Hardware or software resources used to facilitate an activity, particularly an attack.
Note:This is a generic, fallback category. As per its original definition, if a more specific type like 'url', 'ipv4-addr', or 'domain-name' can be used, it should be preferred. 'infrastructure' is for cases where the type is ambiguous or is a higher-level concept (e.g., describing "the attacker's C2 network" as a whole).
Example:Attacker C2 Server

###Main Topic:Data and Credentials###
Name:'credential-value'
Definition:A piece of secret information used for authentication, such as a password, access token, or API key.
Note:This is a specific data type, distinct from 'user-account', which represents the account entity itself. For security reasons, the actual secret value should rarely be stored directly; this entity often represents the existence or type of credential that was compromised.
Example:Stolen Kerberos Ticket

Name:'x509-certificate'
Definition:Refers to a digital certificate that conforms to the X.509 standard, which verifies the authenticity of a public key by binding it to an identity (such as an individual, an organization, or a domain name).
Note:This object is very important when describing cybersecurity incidents, especially in TLS/SSL encrypted communications and code signing. Attackers may use self-signed, stolen, or forged certificates to encrypt C2 communications or to make malware appear legitimate. Its key properties include the issuer, subject, serial number, and various hashes of the certificate (e.g., SHA-1, SHA-256). The hash is often used as the unique identifier for the node.
Example:50D4583B1B35391AA89E86148B267974937447BF

###Main Topic:Intelligence, Defense, and Analysis###
Name:'indicator'
Definition:A pattern of observables or properties that signifies malicious or suspicious activity, serving as a forensic or detection artifact.
Note:An indicator is not the raw data itself, but rather an analytical conclusion or pattern derived from it. It's a critical distinction: An IP address is just an 'ipv4-addr' observable; that same IP address, when known to be malicious and packaged with context (e.g., "C2 server for Emotet"), becomes an 'indicator'. It's the bridge between raw data and actionable intelligence.
Example:[file:hashes.'MD5' = 'd41d8cd98f00b204e9800998ecf8427e']

Name:'course-of-action'
Definition:A recommended step or set of steps to mitigate a threat or remediate an incident.
Note:This entity represents a recommended defensive action, not a technical object itself. It is the answer to "What should we do about this threat?". It is the direct counterpoint to an 'attack-pattern', and is often linked to a threat via the 'mitigates' relationship.
Example:Apply patch for CVE-2021-44228

Name:'security-product'
Definition:A commercial or open-source software, hardware, platform, or service designed to provide cybersecurity functions.
Note:This should be distinguished from 'hacker-tool' (which is typically offensive) and 'general-software' (whose primary purpose is not security). Some tools, like network analyzers, can be dual-use, but are categorized here when used for defense.
Example:CrowdStrike Falcon

Name:'malware-analysis-document-or-publication-or-conference'
Definition:A specifically named or titled source of cybersecurity information, which can be a specific document, a formal publication, or an event.
Note:The key characteristic of this type is that the entity must have a specific, unique name or title. This rule is what distinguishes it from a 'generic-noun' like "Threat Report". For example, "Mandiant APT1 Report" is a specific instance and belongs here, while the term "Threat Report" itself is a 'generic-noun'. Intelligence is often derived from or 'based-on' entities of this type.
Example: 'Mandiant APT1 Report','Kaspersky Lab's "Operation Aurora" White Paper',

###Main Topic:Geographic and Contextual Information###
Name:'location'
Definition:A geographical location, which can range from a country to a specific street address.
Note:This entity is used to provide geographical context to other entities. It can be linked to a 'threat-actor-or-intrusion-set' via the 'originates-from' relationship or to an 'infrastructure' object via the 'located-at' relationship to specify its physical presence.
Example:Beijing, China

###Main Topic:Abstract and Fallback Categories###
Name:'abstract-concept'
Definition:Describes high-level, typically uncountable, ideas, fields of study, principles, or broad categories of activity that do not represent a specific, individual entity.
Note:This type is specifically for uncountable nouns that represent broad ideas. It serves as a high-level fallback category, following the priority: a specific entity type (e.g., 'malware') -> 'abstract-concept' -> 'other' -> 'noise'. This distinguishes abstract ideas from both concrete entities and countable categories of things ('generic-noun').
Example:Cyber Crime, Ransomware Ecosystem, Incident Response, Geopolitics.

Name:'generic-noun'
Definition:A term that refers to a class or type of entity, typically a countable noun, rather than a specific, named instance of that entity.
Note:This type is specifically for countable nouns that represent a class of objects. It helps differentiate between the name of a category and a specific member of that category. It functions within the classification fallback hierarchy: a specific entity type (e.g., 'malware') -> 'generic-noun' -> 'other'-> 'noise'.
Example:Threat Report (as a word), white paper (as a word), Malware (as a word), Vulnerability (as a word).

Name:'other'
Definition:Any other valid entities related to cyber threats that do not fit into any of the other defined categories.
Note:This category functions as a catch-all for meaningful data that is currently unclassifiable. It serves as a source for identifying potential new entity types as the classification schema evolves.
Example:Anti-Ransomware Day, MITRE ATT&CK

Name:'noise'
Definition:Data that is unidentifiable, incorrectly formatted, nonsensical, or otherwise irrelevant junk.
Note:This type is used to flag and isolate poor quality or corrupted data, preventing it from polluting analysis. It is the final classification for data that has no informational value.
Example:'quality#@!', ' (9/1718) [Japanese]'.
'''


grid_rel_type_definition='''
###Main Topic:Attack and Compromise Relationships###
Name:'exploits'
Definition:Represents an entity leveraging a specific flaw or weakness within another entity (typically a Vulnerability) to achieve a malicious objective.
Note:This is a more specific form of 'uses'. When the action involves taking advantage of a known vulnerability, 'exploits' should be used instead of 'uses'.
Example:The malware exploits the CVE-2021-44228 vulnerability.

Name:'bypasses'
Definition:Indicates that an offensive entity (e.g., malware, exploit) successfully evades or circumvents a defensive measure. If 'mitigates' (such as patching to mitigate vulnerabilities) are successful actions for defenders, then 'bypasses' (such as using obfuscation techniques to bypass sandboxes) are successful actions for attackers. It is specifically used to describe the behavior of circumventing and bypassing defensive measures.
Example:The malware's obfuscation technique bypasses sandbox analysis.

Name:'malicious-investigates-track-detects'
Definition:Represents a malicious action where one entity (typically malware or a tool) performs either a discrete investigation, continuous tracking, or active detection of another entity to gather information or for evasive purposes.
Note:This relationship now covers three types of malicious information gathering and reconnaissance: Investigating: One-time reconnaissance of an entity (e.g., a system scan). Tracking: Long-term, continuous surveillance of an entity (e.g., keystroke logging). Detecting (Malicious): Evasion-focused discovery, such as identifying a sandbox, debugger, or specific security tool to alter behavior.
Example:Example 1 (Investigates): A malware implant malicious-investigates-track-detects local system configuration files. Example 2 (Tracks): A spyware module malicious-investigates-track-detects the user's web Browse history. Example 3 (Detecting): The malware malicious-investigates-track-detects the presence of a virtual machine environment.

Name:'impersonates'
Definition:Indicates that one entity actively masquerades as another, distinct entity to deceive or gain trust.
Note:This is distinct from 'alias-of'. 'impersonates' is a deceptive action between two separate entities. In contrast, 'alias-of' links two different names for the very same entity. For example, a hacker 'impersonates' the CEO in an email, whereas "APT28" is an 'alias-of' "Fancy Bear."
Example:A threat actor impersonates a trusted IT administrator to trick users.

Name:'targets'
Definition:Describes an offensive entity directing its actions against another entity. It expresses the intent and direction of an attack.
Note:'targets' describes intent, while 'compromises' describes a successful outcome. An actor might 'target' the financial industry for years but 'compromise' a specific bank in a single operation. 'targets' is also broader than 'exploits'; an actor can 'target' an organization, whereas they 'exploit' a specific vulnerability within that organization's systems.
Example:A phishing campaign targets employees in the financial sector.

Name:'compromises'
Definition:Represents that an offensive entity has successfully violated the confidentiality, integrity, or availability of a target, achieving some form of unauthorized access or control.
Note:See the note under 'targets' for a direct comparison.
Example:The threat actor compromised the company's domain controller.

Name:'leads-to'
Definition:Describes a causal relationship where one entity or event directly results in another outcome or state, often used in attack chains. Relationships such as 'exploits', 'delivers', and 'executes' are all “points” in the attack chain, and 'leads-to' is the “line” connecting these points, clearly showing the logic of “vulnerability exploit leads to remote code execution”. When a relationship meets the subdivision relationship of 'exploits', 'delivers', and 'executes', choose them instead of 'leads-to'.
Example:Exploitation of a vulnerability leads-to remote code execution.

###Main Topic:Data and Payload Movement###
Name:'drops'
Definition:Represents an entity creating a new file on the local filesystem from its own embedded or internal resources.
Note:This relationship exclusively describes the action of Local -> Local file creation, with no network communication involved. This is distinct from 'downloads', which is an External -> Local action.
Example:The installer drops a malicious DLL file into the System32 folder.

Name:'downloads'
Definition:Represents an entity retrieving a file or data from an external, remote source and saving it to the local system.
Note:This relationship exclusively describes the action of External -> Local data transfer. It is the direct opposite of 'drops', which involves no network communication.
Example:The dropper downloads a second-stage payload from a malicious URL.

Name:'executes'
Definition:Signifies that one entity (e.g., a loader, script) runs or initiates another entity (e.g., a malicious executable).
Example:A dropper executes a second-stage payload.

Name:'delivers'
Definition:Represents a higher-level, abstract relationship where one attack component is responsible for 'bringing' a malicious payload or tool to the target environment.
Note:This describes the abstract "bringing" action within an attack chain, answering "How did the payload get here?" at a tactical level. For example, a phishing email 'delivers' malware; this delivery might be achieved through the user 'downloads' an attachment, which then 'drops' an executable.
Example:A phishing campaign delivers the Ursnif malware.

Name:'beacons-to'
Definition:Specifically indicates that malware or an implant periodically sends 'beacon' or 'heartbeat' signals to its Command and Control (C2) server.
Example:Malware beacons-to (beacons-to) Command and Control URL.

Name:'exfiltrate-to'
Definition:Specifically describes the act of stealing data from a compromised system and transmitting it outward to a target location specified by the attacker, such as a server or IP address.
Note:The core of this relationship is purposeful, outbound data transmission. Its distinction from other network relationships lies in intent and direction: (1) Versus 'communicates-with': 'exfiltrate-to' is a specific type of 'communicates-with'. If the purpose of the communication is confirmed to be data theft, 'exfiltrate-to' should be preferred for more precise semantics. If the purpose is unknown, the more general 'communicates-with' should be used. (2) Versus 'downloads': The data flow direction is the opposite of 'downloads'. 'downloads' refers to fetching files from an external source into the victim system, while 'exfiltrate-to' refers to uploading data from the victim system to an external source. (3) Versus 'leaks': 'exfiltrate-to' typically describes a targeted, covert transfer from a victim to an attacker. In contrast, 'leaks' (if used as a custom relationship) usually refers to a broader, potentially public or semi-public data disclosure.
Example:A spyware implant (Malware) exfiltrate-to a specific FTP server (Infrastructure) to upload stolen documents.

Name:'leaks'
Definition:Represents the unauthorized disclosure or public release of sensitive resources. This includes confidential data (e.g., documents, credentials) as well as operational assets like malware source code or vulnerability details. 
Note:'exfiltrate-to' (malware steals data to a server) describes the directed transfer of data from the victim to the attacker. 'leaks' (internal threat actors leak company documents) describes the unauthorized public or semi-public disclosure of sensitive resources (data, source code, etc.). The core difference between the two lies in the direction of information flow and the degree of disclosure.
Example:An insider threat leaks confidential corporate documents online. The source code for a prominent banking trojan leaks onto a public repository.

Name:'communicates-with'
Definition:Describes the occurrence of network communication between two entities. It is a general relationship for network interactions.
Note:'beacons-to', 'downloads', and 'exfiltrate-to' are all specific types of 'communicates-with'. If the traffic is a periodic heartbeat, 'beacons-to' is more precise. If the purpose is to retrieve a file, use 'downloads'. If it is to send data out, use 'exfiltrate-to'. Use 'communicates-with' for general descriptions or when the specific purpose is unknown.
Example:The implant communicates-with a C2 server every hour.

###Main Topic:Infrastructure and Provisioning###
Name:'resolves-to'
Definition:A specific technical relationship describing a domain name being resolved to one or more IP addresses via the Domain Name System (DNS).
Note:See the note under 'hosts'. This relationship is a core technical link for establishing network infrastructure associations.
Example:The malicious domain https://www.google.com/search?q=evil-phishing.com resolves-to the IP address 198.51.100.10.

Name:'hosts'
Definition:Indicates that an infrastructure entity 'carries' or provides the runtime environment for another object, such as a malicious payload, website, or C2 service.
Note:This relationship describes 'carrying' at the infrastructure level. It is distinct from 'delivers', which describes a tactical action, and 'provides', which is more general. For example, a server ('hosts') a malware file, which is then ('downloads') by a victim after being ('delivers') by a phishing link.
Example:A bulletproof hosting provider hosts malware command and control servers.

Name:'provides'
Definition:A general relationship where one entity supplies another with a resource, service, or capability.
Note:This is the most abstract supply relationship and should be used when a more specific term is not applicable. Follow the priority: use 'delivers' for tactical delivery or 'hosts' for infrastructure hosting first. Use 'provides' only when the relationship is more general than these options.
Example:A bulletproof hosting service provides infrastructure for a phishing campaign.

###Main Topic:Attribution and Association###
Name:'authored-by'
Definition:Defines the creator or development source of an entity, such as malware, a tool, a report, or an attack pattern. It is used to trace the provenance of an object.
Note:The core of this relationship is to clarify "who created it". It has key distinctions from other relationships: (1) Versus 'attributed-to': 'authored-by' focuses on the act of creation itself, while 'attributed-to' focuses on assigning responsibility for an attack campaign. An organization can have 'authored-by' a tool, while the campaign that uses the tool is 'attributed-to' another group. (2) Versus 'owns': 'owns' describes the state of ownership over infrastructure or tools, while 'authored-by' describes their creation source.
Example:The Lazarus Group (Identity) authored-by a custom backdoor malware (Malware).

Name:'owns'
Definition:Describes a real-world entity (e.g., an organization, team, or an individual) having ownership or de facto dominion over another entity (e.g., infrastructure, a domain name, or a tool).
Note:The core of this relationship is ownership by a real-world entity. Its distinction from 'controls' lies in the nature of the subject: the subject of 'owns' is a real-world entity (a team, an individual), while the subject of 'controls' is software. This is a critical distinction as it separates the real-world actor from their digital-world proxy tools.
Example:The APT41 group (Identity) owns the domain name evil-domain.com and the C2 server.

Name:'controls'
Definition:Specifically describes the relationship where one software entity (e.g., a trojan, backdoor, RAT) commands and controls another software entity (e.g., a hijacked process, a browser plugin).
Note:The core of this relationship is software-level control. Its key distinction from 'owns' is the level of the controller: the subject of 'controls' is a piece of software (e.g., a RAT), while the subject of 'owns' is a real-world entity (e.g., a team). For example, a team can 'own' a domain name, and the RAT program on the C2 server pointed to by that domain then 'controls' another process on the victim host.
Example:A Remote Access Trojan (RAT) controls a compromised browser process to steal cookies.

Name:'attributed-to'
Definition:Formally assigns the responsibility for a threat activity, such as an Intrusion Set or Campaign, to one or more Threat Actors. This is typically the conclusion derived from intelligence analysis and attribution efforts.
Note:It differs from 'authored-by' and 'affiliated-with'. 'attributed-to' focuses on the responsibility for an attack, while 'authored-by' pertains to the creation of an entity, like malware. An organization might 'author' a tool, but if another affiliated group uses it in an attack, the attack activity is 'attributed-to' the latter. 'affiliated-with' describes a broader organizational or social connection (e.g., membership, employment), whereas 'attributed-to' is a specific assignment of culpability for an action.
Example:Intrusion Set "Sandworm" is attributed-to Russian GRU Unit 74455.

Name:'affiliated-with'
Definition:Describes an affiliation, employment, or membership relationship between individuals and organizations. 'authored-by' refers to the creation relationship, 'attributed-to' refers to the responsibility for the attack, and 'owns' refers to the ownership of the infrastructure. 'affiliated-with' describes an 'affiliation' relationship at the organizational or social level, which is not necessarily creation, attack or ownership. When a relationship meets the subdivision relationship of 'attributed-to' and 'owns', choose them instead of 'affiliated-with'.
Example:A security researcher is affiliated-with a university.

Name:'cooperates-with'
Definition:Describes active, non-hierarchical collaboration between two or more peer entities, such as threat groups working together. 'affiliated-with' describes an affiliation. 'cooperates-with' (threat A cooperates with threat B) describes a collaborative relationship between peer entities.
Example:Threat Actor A cooperates-with Threat Actor B in a joint operation.

###Main Topic:Composition, Capability and State###
Name:'is-part-of'
Definition:Used when one entity is a component, member, or constituent of a larger entity. It is the inverse of ''consists-of''.
Example:A malicious module is-part-of a larger malware family.

Name:'consists-of'
Definition:Describes the compositional relationship where a complex entity is made up of its structural subcomponents.
Note:This relationship should be used to detail an object's "bill of materials" or internal architecture. It is distinct from 'has', which is used to attribute abstract features or capabilities rather than constituent parts. Use 'consists-of' to answer the question, "What is it made of?"
Example:The TrickBot malware framework consists-of numerous distinct modules, such as a password grabber and a VNC module.

Name:'has'
Definition:Indicates that an entity possesses a specific feature, function, or capability, which may be abstract in nature.
Note:This relationship is best used for attributing characteristics or functions to an object. It differs from 'consists-of', which is used for deconstructing an object into its physical or logical components. Use 'has' to answer the question, "What can it do?" or "What properties does it possess?"
Example:A backdoor Trojan has a persistence capability.

Name:'depends-on'
Definition:Signifies that one entity requires another entity to exist or function correctly.
Note:This describes a state of prerequisite or dependency. It differs from uses, which describes an action. For example, malware uses PowerShell to execute commands, but it depends-on a specific library to run. It covers terms like requires and is required for.
Example:A malware depends-on a specific version of the .NET Framework.

Name:'creates-or-generates'
Definition:An entity dynamically creates or generates another entity, such as a file, process, or data.
Note:This is more general than authored-by (which is about original creation by an identity) and drops (which is specific to malware placing a file). It describes the runtime action of creation. It covers terms like create, creates, and generates. If the relationship is more concise, such as a malware creating a file, use 'drops' instead, or if it is about the original creation by an identity, use 'authored-by'. Otherwise, use 'creates-or-generates' to capture the action of creation or generation in a broader sense.
Example:A malware creates-or-generates a new registry key. A malware creates-or-generates notification popups.

Name:'modifies-or-removes-or-replaces'
Definition:Indicates that an entity alters, replaces, or removes another entity or its components, such as changing a registry key.
Example:A ransomware modifies(modifies-or-removes-or-replaces) the Master Boot Record.

Name:'uses'
Definition:Represents that an entity employs or leverages another entity to achieve its objectives. It is a highly general, active relationship describing "A uses B to do something."
Note:Differentiated from 'depends-on' and 'exploits'. 'uses' is an active behavior (e.g., malware uses PowerShell to execute commands), while 'depends-on' is a static, prerequisite state (e.g., the malware's execution depends-on the .NET Framework). 'exploits' is a special case of 'uses' that specifically involves leveraging a 'vulnerability'; if a vulnerability is leveraged, 'exploits' should be preferred.
Example:Threat Actor APT41 uses the Cobalt Strike framework.

###Main Topic:Classification and Lineage###
Name:'variant-of'
Definition:Indicates that one entity is a direct evolutionary version of another, typically sharing a lineage in code or core functionality.
Note:This is distinct from 'derived-from' and 'compares-to'. 'variant-of' implies direct derivation, often at the code level (e.g., the Zeus malware has countless 'variants'). 'derived-from' is more abstract, signifying conceptual or technical inspiration without direct code reuse. 'compares-to' is for a general comparison of attributes without implying any lineage.
Example:The Gootkit malware is a variant-of the earlier Gozi trojan.

Name:'derived-from'
Definition:Indicates that an entity is conceptually, technically, or philosophically inspired by or based on another, but is not a direct code-level evolution.
Note:See the note under 'variant-of'. 'derived-from' represents a more abstract, "intellectual lineage" relationship.
Example:The techniques used in the Triton malware were derived-from the know-how developed for the Stuxnet attack.

Name:'alias-of'
Definition:Indicates that one entity is an alternative name or identifier for another.
Note:This provides a direct and explicit way to link known aliases, which is more specific than the broader compares-to relationship. It is a bidirectional relationship. This covers terms like has alias and is alias of.
Example:APT28 alias-of Fancy Bear.

Name:'compares-to'
Definition:Indicates a comparative relationship between two entities based on their features, behavior, complexity, or other attributes. 'variant-of' means two entities have a direct evolution or code-derived variant relationship, while 'compares-to' is broader and can include any form of comparison. When a relationship meets both criteria, 'variant-of' should be used instead.
Example:Malware A compares-to Malware B in its propagation method.

Name:'categorized-as'
Definition:Links an entity to its formal classification or type within a given taxonomy. 'variant-of' is a specific evolutionary classification. 'categorized-as' is a more formal, ontological classification relationship, for example, used to link an instance to a category in a taxonomy.
Example:The threat activity is categorized-as a form of ransomware attack.

###Main Topic:Geographic Relationships###
Name:'located-at'
Definition:Specifies the current or known geographic location of an entity.
Note:This is distinct from 'originates-from'. 'located-at' refers to the present location, while 'originates-from' refers to the place of origin or provenance. For example, a threat actor may 'originates-from' Iran, but the server they use is 'located-at' a data center in the Netherlands.
Example:A command and control server is located-at a data center in Germany.

Name:'originates-from'
Definition:Specifies the place of origin or provenance of an entity.
Note:See the note under 'located-at' for a direct comparison.
Example:The Stuxnet malware is believed to originate-from the United States and Israel.

###Main Topic:Analysis and Defense Relationships###
Name:'indicates'
Definition:Represents an inferential relationship where the presence of one entity (typically an Indicator) serves as evidence or a sign of another threat entity. It expresses that "if A is observed, it likely signifies that B exists or is occurring."
Note:The core of this relationship is analytical inference. It is distinct from the 'detecting' function within other relationships (e.g., 'research-describes-analysis-of-characterizes-detects'). The 'detecting' function represents an active, confirmed discovery, whereas 'indicates' represents a probabilistic link ("this likely means that"). This relationship is fundamental for operationalizing threat intelligence, as it directly connects a detectable artifact (the IOC) to the threat it helps to identify.
Example:An IP address (indicator) indicates a malware.

Name:'mitigates'
Definition:Indicates that a defensive measure or Course of Action effectively counters, reduces, or remediates the threat posed by an Attack Pattern, Vulnerability, or Malware.
Note:This is the inverse of 'bypasses'. 'mitigates' is a successful action for the defender (e.g., a patch mitigates a vulnerability), whereas 'bypasses' is a successful action for the attacker (e.g., an obfuscation technique bypasses a sandbox).
Example:Applying the MS17-010 patch mitigates the EternalBlue exploit.

Name:'based-on'
Definition:Indicates that an object (e.g., report, indicator, signature) is derived from or based on the information or analysis of another object (e.g., observed data, another report, malware sample).
Example:Indicator based-on (based-on) Observed Data.

Name:'research-describes-analysis-of-characterizes-detects'
Definition:A comprehensive research and defense relationship that signifies a document describing a subject, an actor analyzing a subject, a formal analysis object characterizing a subject's behavior, or a defensive tool identifying a threat.
Note:This consolidated relationship serves four primary purposes:Describing: Linking a textual document or publication to the entity it is about.Analyzing: Linking an analytical actor (e.g., a researcher or organization) to the subject of their investigation.Characterizing: Linking a formal analysis object (e.g., a Malware Analysis run) to the entity it was performed on.Detecting (Defensive): Linking a defensive tool, signature, or security product to the threat it successfully identifies.
Example:Example 1 (Describing): A Mandiant report research-describes-analysis-of-characterizes-detects the APT1 group.Example 2 (Analyzing): A security researcher research-describes-analysis-of-characterizes-detects a new malware sample. Example 3 (Characterizing): A sandbox analysis run research-describes-analysis-of-characterizes-detects the WannaCry malware. Example 4 (Detecting): An antivirus signature research-describes-analysis-of-characterizes-detects a specific malware file.

###Main Topic:Meta and Fallback Relationships###
Name:'negation'
Definition:Represents the confirmed absence of a relationship, link, characteristic, or action between entities.
Note:This type is used to explicitly state that a suspected or potential relationship does not exist. It is crucial for refuting claims or clarifying the scope of an entity's attributes. It should be used for phrases like does not contain, has no links to, is not affected by.
Example:A threat report states that Malware X negation (is not affected by) Vulnerability Y.

Name:'other'
Definition:If a relationship exists but does not fit into the categories above, and write down the value of 'rel' as the original text of the relationship.
Example: Not available, Not Applicable, Unknown, etc.
'''


def grid_kg_single_prompt_maker_tracerawtext_20260303(text): 
    text = str(text)
    
    prompt_message = [
        {
            'role': 'user',
            'content': (
            '''
            You are an NLP model specialized in threat intelligence extraction. Your task is to extract a knowledge graph related to cybersecurity threats from a given threat intelligence report or blog article, including entities (nodes) and relationships (edges) among those entities, and output the results in a specified format.

            [FIX-260303-1] ⚠️ OVERRIDING CONSTRAINT: Text-Provable Truth
            All extraction MUST be grounded in direct textual evidence. The following are FORBIDDEN:
            - External knowledge completion: Do NOT infer relationships using domain knowledge not stated in the text.
            - Subject elevation: Do NOT attribute a tool/component's behavior to its controller/parent entity unless the text explicitly states so.
            - Behavioral-to-structural conversion: Do NOT convert "A uses B" into "B is-part-of A" unless the text explicitly states a structural composition relationship.
            - Chain deduction with subject change: If text says "A uses B" and "B does C", do NOT extract "A does C". Instead, extract (A, uses, B) and (B, does, C) as separate edges.
            When in doubt, preserve only directly text-supported relations.

            [Entities (Nodes)]
            What's an entity:

            In cyber threat intelligence, 'entity' refers to any unit of information that can be independently identified, described, and analyzed, and it forms a fundamental component of each link in threat activity. It is important to note that entities here are not limited to those verified indicators (IOCs) specifically used for detection, such as specific IP addresses, domain names, or file hashes, but a broader concept. Entities can be objects with clear names and characteristics (such as 'get-logon-history.ps1'), or data objects such as 'RAR file', even if it has no fixed naming rules.

            The main characteristics of entities are:

                Independence: Each entity exists as an independent unit of information and can be extracted and analyzed separately;

                Relevance: There may be inherent connections between entities, and by associating this information, a complete attack chain or threat portrait can be constructed;

                Diversity: Entities can cover various forms of information such as files, scripts, configuration files, registry entries, network traffic data, log records, etc.;

                Contextual significance: Even if an entity (such as a downloaded 'RAR file') does not have a specific name, as long as it has contextual significance and analytical value in threat intelligence analysis, it is still a valid entity.

            Important Note

                Only extract entities from named entities that directly appear in the current input text, i.e., only pay attention to the entities explicitly mentioned in the current input text.

            Entity Extraction Steps
            Stage 1 – Entity Extraction and Classification

            Stage 1.1 Fully scan the #current input's text# to identify all entities related to cybersecurity threats, including:

                Explicitly named indicators (IPs, domains, filenames, hash values, etc.)

                Implicit threat components (attack stages, undocumented tools, generic file types)

                Contextually significant objects ('RAR file', 'registry entry') even without specific names

            Stage 1.2 – Type Assignment

                Assign one predefined category to each entity.

                Use threat-actor/intrusion-set formatted names for unnamed attackers/attacks.

                Apply other category only when no predefined type matches.

            Stage 1.3 – Primary Name and Alias Handling
                This is a critical step for ensuring consistency.
                1.  **Primary Name (`name` field):** The `name` attribute for an entity MUST be the specific string used when that entity is **first identified** in the text.
                    * Example: If the text first says "We discovered **DarkPanda malware**...", the `name` must be 'DarkPanda malware'.
                2.  **Alias Recording (`alias` field):** The `alias` field MUST capture **all other subsequent mentions or alternative references** to that same entity found in the text. This is not just for formal names.
                    * **Includes Formal Aliases:** e.g., 'APT28' (name) and 'Fancy Bear' (alias).
                    * **Includes Generic/Pronominal References:** e.g., 'the malware', 'that malware', 'this specific tool', 'the script'.
                    * **Includes Abstract Co-references (MANDATORY):** You must resolve abstract co-references. If an abstract phrase refers back to a previously mentioned entity, it MUST be included. For example, if the `name` is 'a genuine social media account', a later reference to it as '**the real thing**' MUST be captured in the `alias` list as 'the real thing'.
                3.  **Lineage (`mother entity` field):** Identify evolutionary relationships for the 'mother entity' field (e.g., malware variants).

            Stage 1.4 – Recheck (Self-Reflection on Completeness)
            Before finalizing entity extraction, pause and critically reflect on whether any relevant entities may have been overlooked.
            First, check if any explicitly named entities were unintentionally ignored.
            Then, examine whether non-explicitly named entities have been missed—these may be referenced using generic terms like file, image, or script, but actually point to specific entities in context.
            
            ''' + grid_entity_type_definition + '''

            Stage 2 – Relationship Extraction and Classification
            Your relationship extraction process will follow a comprehensive two-pass approach. First, you will identify the primary action-based relationship. Second, you will identify all structural, definitional, and descriptive relationships.

            Stage 2.1 – Pass 1: Primary Action Relationship Extraction
            To determine the main action relationship between entities, you MUST follow the strict, prioritized decision process below. Evaluate these cases in order and stop as soon as one case is met for the sentence's primary verb.
            Case 1: An Explicit Action Verb Phrase Exists in the Text (Highest Priority)
            - Check for an explicit action verb (e.g., 'deploys', 'extracts', 'communicates-with').
            - If YES: The 'rel' value MUST BE this explicit phrase, and 'rel_type' must be the best match from the predefined list. If no good match exists, 'rel_type' MUST be ['other'].
            - Example: "LazyFox 'deployed' Cobalt Strike." -> { 'sub': 'LazyFox', 'rel': 'deployed', 'rel_type': ['uses'], 'obj': 'Cobalt Strike' }

            Case 2: No Explicit Verb, but an Implied Action Matches a Predefined rel_type
            - If Case 1 does not apply, check if the implied relationship perfectly matches a predefined 'rel_type'.
            - If YES: Both 'rel' and 'rel_type' MUST BE the name of that matching category.
            - Example: "...traffic between the host and evil.com." -> { 'sub': 'host', 'rel': 'communicates-with', 'rel_type': ['communicates-with'], 'obj': 'evil.com' }

            Case 3: No Explicit Verb and No Predefined rel_type Fits (Fallback)
            - If neither Case 1 nor Case 2 applies, infer the underlying action, summarize it concisely as the 'rel' value.
            - The 'rel_type' in this case MUST BE ['other'].
            - Example: "The attacker stored credentials inside an encoded string." -> { 'sub': 'attacker', 'rel': 'hides credentials in', 'rel_type': ['other'], 'obj': 'encoded string' }

            Stage 2.1.1 – Pass 2: Structural and Definitional Relationship Extraction
            After completing Pass 1, you MUST re-scan the text to extract all additional relationships defined by the following grammatical structures, even if they do not contain a primary action verb. Extract all that apply.
            
            1.  Possessive Relationships (`of`, `'s`):
                -   Structure: `Y of Z` or `Z's Y`.
                -   Action: This often implies composition or ownership. You MUST extract a relationship. Use `rel_type` like ['is-part-of'], ['owns'], or ['has'].
                -   Example: "the memory 'of' the browser" -> `{'sub': 'memory', 'rel': 'is-part-of', 'rel_type': ['is-part-of'], 'obj': 'browser'}`.
                -   Example: "the actor's toolkit" -> `{'sub': 'actor', 'rel': 'owns', 'rel_type': ['owns'], 'obj': 'toolkit'}`.

            2.  Prepositional Phrases of Location/Containment (`in`, `at`, `on`, `within`, etc.):
                -   Structure: `Y in Z`, `Y at Z`, `Y inside Z`.
                -   Action: This implies location. You MUST extract a `located-at` relationship.
                -   Example: "a malicious DLL 'in' the System32 folder" -> `{'sub': 'malicious DLL', 'rel': 'located-at', 'rel_type': ['located-at'], 'obj': 'System32 folder'}`.

            3.  Appositives (Defining one entity as another):
                -   Structure: `Y, a Z`, `Y (also known as Z)`, `Y, the infamous Z`.
                -   Action: This implies alias or categorization. You MUST extract an `alias-of` or `categorized-as` relationship.
                -   Example: "APT28, also known as Fancy Bear, ..." -> `{'sub': 'APT28', 'rel': 'alias-of', 'rel_type': ['alias-of'], 'obj': 'Fancy Bear'}`.
                -   Example: "Cobalt Strike, a post-exploitation tool, ..." -> `{'sub': 'Cobalt Strike', 'rel': 'categorized-as', 'rel_type': ['categorized-as'], 'obj': 'post-exploitation tool'}`.

            4.  Relative Clauses (`which`, `that`):
                -   Structure: `Y, which [verb phrase] Z`.
                -   Action: The clause describes a capability or action of Y. You MUST extract this relationship using the verb from the clause.
                -   Example: "...a script 'that downloads' a payload." -> `{'sub': 'script', 'rel': 'downloads', 'rel_type': ['downloads'], 'obj': 'payload'}`.

            5.  Punctuation-based Lists (Colon, Dash, Parentheses):
                -   Structure: `Y: Z1, Z2, ...` or `Y (including Z1, Z2)`.
                -   Action: This implies composition or attribution. You MUST extract a relationship like `consists-of` or `has` for each item in the list.
                -   Example: "C2 servers: 10.0.0.1, 10.0.0.2" -> Extract TWO relationships: `{'sub': 'C2 servers', 'rel': 'consists-of', 'rel_type': ['consists-of'], 'obj': '10.0.0.1'}` AND `{'sub': 'C2 servers', 'rel': 'consists-of', 'rel_type': ['consists-of'], 'obj': '10.0.0.2'}`.
            
            [FIX-260303-2] Stage 2.2 – Recheck (Text-Provable Constraint)
            Ensure that every entity extracted in Stage 1 has at least one corresponding relationship that is DIRECTLY SUPPORTED by the text.
            If an entity has no text-supported relationship, do NOT fabricate or infer an indirect relationship using external knowledge. Instead, REMOVE that entity from the entity list — an entity without text-provable relationships should not be included in the output KG.
            ⚠️ FORBIDDEN: Do NOT infer relationships like "Entity X might be related to APT Y because it appeared in the same report". Such inference is external knowledge completion, not text-provable extraction.

            Stage 2.3 – Factuality Annotation for Relationships
            For each extracted relationship, determine its factuality status by examining the sentence context:
            - Scan for modal verbs, negation words, or temporal indicators that affect the relationship's certainty.
            - The 'special_factuality' field is MANDATORY and must ALWAYS be a list (array), even when empty or containing a single value.
            - Multiple factuality markers can coexist in the same relationship. Evaluate ALL applicable markers and include them in the list.
            - Assign 'special_factuality' values according to these rules:
                1. If the relationship is negated or denied → add 'negated' to the list
                2. If the relationship describes future intent or planned action → add 'future' to the list
                3. If the relationship is described with uncertainty → add 'possible' to the list
                4. If none of the above apply (stated as fact) → use [''] (list with empty string)
                5. Important: Multiple markers can coexist - do NOT prioritize, include ALL that apply.
            
            Examples:
            - "The actor may deploy Cobalt Strike." → special_factuality: ['possible']
            - "The group will target banks." → special_factuality: ['future']
            - "No evidence indicates C2 communication." → special_factuality: ['negated']
            - "The malware likely is a variant of Emotet." → special_factuality: ['possible']
            - "The actor might plan to use this exploit." → special_factuality: ['possible', 'future']
            - "The malware communicates with the server." → special_factuality: ['']
            
            ''' + grid_rel_type_definition + '''

            Stage 3 – Output Generation
            Both the entity list and the relationship list must strictly follow the formats below. Ensure the entity names are consistent in both the entity list and the relationships, and use correct JSON formatting:
            
            *** CRITICAL FORMATTING RULE: WHITESPACE PRESERVATION ***
            For all fields starting with "raw_" (raw_sub_name, raw_obj_name, raw_text_start, raw_text_end), you MUST preserve the original text's whitespace EXACTLY.
            1. **Newlines (`\\n`):** If the span of text in the original document crosses a line break, your JSON string MUST include the `\\n` character (or `\\r\\n`). Do NOT replace newlines with spaces.
            2. **Multiple Spaces:** If the original text contains multiple consecutive spaces (e.g., "User   Account"), you MUST reproduce all of them. Do NOT collapse them into a single space.
            3. **Tabs:** Preserve tabs (`\\t`) exactly as they appear.
            The goal is that a strict string search (content.find()) for your extracted value must return the exact index, not -1.

            Part 1: Entity List

                The entire JSON array must be strictly enclosed between #Entity_List_Start# and #Entity_List_End#.

                Each entity node must include the following attributes (all attribute values should be strings or string arrays):

                    name: The specific name of the entity, using the string from its **first appearance** in the text.

                    type: The category of the entity (each entity node must have only one type value).

                    alias: A list of **all other strings** used to refer to this entity in the text.
                        This includes formal aliases (e.g., 'Fancy Bear') and generic references (e.g., 'the malware', 'that script').
                        If multiple aliases exist, format as ['Alias1', 'Alias2'].
                        If the entity is only ever referred to by its primary 'name' and has no other mentions, use ['None'].

                    mother entity: If the entity is a variant or evolution of another entity, provide the name of its parent entity; otherwise, use ['None'].

            Part 2: Entity Relationships

                Extract relationship descriptions between entities from the text and output them as a JSON array of objects.
                The entire JSON array must be strictly enclosed between #Relationship_List_Start# and #Relationship_List_End#.

                Each relationship object must follow this format:

                {
                    'sub': '<Source Entity Standardized Name>',
                    'raw_sub_name': '<Verbatim Sub in Text>',
                    'rel': '<Relationship Text>',
                    'rel_type': ['<Relationship Type Category>'],
                    'obj': '<Target Entity Standardized Name>',
                    'raw_obj_name': '<Verbatim Obj in Text>',
                    'raw_text_start': '<Starting words of the raw text evidence>',
                    'raw_text_end': '<Ending words of the raw text evidence>',
                    'special_factuality': '<Factuality Status>'
                }

                Field Definitions:

                    sub: Must exactly match the source entity `name` extracted in Part 1.

                    **raw_sub_name (Search Anchor):** The specific string representing the Subject in this specific relationship context. 
                        - **CRITICAL CONSTRAINT (Verbatim & Unique):** This string MUST be the exact, verbatim text found in the document.
                        - **WHITESPACE FIDELITY:** If the text spans across a line break, include the `\\n`. If it has double spaces, keep them.
                        - **Uniqueness:** If the entity name (e.g., "APT28" or "the tool") appears multiple times, you **MUST** extend the string by including adjacent words (preceding or succeeding words) until this specific instance is unique in the whole text.
                        - Example: Instead of just "the tool" (which appears 5 times), use "Finally, the tool" or "the tool automatically".

                    rel: A verb or phrase summarizing the relationship as described in the text.

                    rel_type: An array listing one or more of the predefined relationship types.

                    obj: Must exactly match the target entity `name` extracted in Part 1.

                    **raw_obj_name (Search Anchor):** The specific string representing the Object in this specific relationship context.
                        - **CRITICAL CONSTRAINT (Verbatim & Unique):** Same as above. You MUST extend the string (e.g., include prepositions or verbs like "to the server" or "deployed malware") to ensure it is unique and searchable.
                        - **WHITESPACE FIDELITY:** Preserve `\\n` and multiple spaces strictly.

                    raw_text_start: **MANDATORY.** The starting word(s) of the verbatim text snippet from the original input that provides the evidence for the relationship.
                        - **Warning:** Do NOT trim trailing spaces or newlines if they are part of the unique starting sequence you selected.

                    raw_text_end: **MANDATORY.** The ending word(s) of the *same* verbatim text snippet.
                    
                            **CRITICAL ANCHOR CONSTRAINTS:**
                            1.  **Uniqueness (Primary):** The span of text in the original content *starting* with `raw_text_start` and *ending* with `raw_text_end` **MUST BE UNIQUE**.
                            2.  **Conciseness (Secondary):** While maintaining uniqueness, the anchors should be as concise as possible.
                            3.  **Whitespace:** If `raw_text_end` falls on a newline or includes punctuation followed by a newline, include it exactly.

                    special_factuality: A list (array) indicating the factuality status.

            Below is an example:

            #Entity_List_Start#
            ```json
            [
            { 'name': 'exampleAPT', 'type': 'threat-actor', 'alias': ['The group'], 'mother entity': ['None'] },
            { 'name': 'exampleTool', 'type': 'hacker-tool', 'alias': ['customized tool'], 'mother entity': ['None'] },
            { 'name': 'exampleCVE', 'type': 'vulnerability', 'alias': ['None'], 'mother entity': ['None'] }
            ]
            ```
            #Entity_List_End#

            #Relationship_List_Start#
            ```json
            [
            {
                'sub': 'exampleAPT',
                'raw_sub_name': 'However, The group',
                'rel': 'utilized',
                'rel_type': ['uses'],
                'obj': 'exampleTool',
                'raw_obj_name': 'a customized\\n tool',
                'raw_text_start': 'possibly utilized',
                'raw_text_end': 'utilized a customized\\n tool',
                'special_factuality': ['possible']
            },
            {
                'sub': 'exampleTool',
                'raw_sub_name': 'Finally, it',
                'rel': 'is using',
                'rel_type': ['exploits'],
                'obj': 'exampleCVE',
                'raw_obj_name': 'exampleCVE specifically',
                'raw_text_start': 'is using',
                'raw_text_end': 'using exampleCVE specifically',
                'special_factuality': ['']
            }
            ]
            ```
            #Relationship_List_End#
            【Start of the full text】
            ''' + 'Now, my input text for your task is: ' + str(text)
            )
        }
    ]
    return prompt_message


def grid_kg_single_prompt_maker_very_simple_20260303(text): 
    text = str(text)

    entity_types = (
        "user-account, identity, threat-actor-or-intrusion-set, "
        "malware, hacker-tool, general-software, detailed-part-of-malware-or-hackertool, detailed-part-of-general-software, "
        "vulnerability, attack-pattern, campaign, "
        "file, process, windows-registry-key, "
        "ipv4-addr, ipv6-addr, domain-name, url, email-address, network-traffic, mac-address, infrastructure, "
        "credential-value, x509-certificate, "
        "indicator, course-of-action, security-product, malware-analysis-document-or-publication-or-conference, "
        "location, abstract-concept, generic-noun, other, noise"
    )

    rel_types = (
        "exploits, bypasses, malicious-investigates-track-detects, impersonates, targets, compromises, leads-to, "
        "drops, downloads, executes, delivers, beacons-to, exfiltrate-to, leaks, communicates-with, "
        "resolves-to, hosts, provides, "
        "authored-by, owns, controls, attributed-to, affiliated-with, cooperates-with, "
        "is-part-of, consists-of, has, depends-on, creates-or-generates, modifies-or-removes-or-replaces, uses, "
        "variant-of, derived-from, alias-of, compares-to, categorized-as, "
        "located-at, originates-from, "
        "indicates, mitigates, based-on, research-describes-analysis-of-characterizes-detects, "
        "negation, other"
    )

    # Note: Whitespace is minimized and Markdown symbols (#, *, -) are removed to save tokens while preserving logic.
    prompt_message = [
        {
            'role': 'user',
            'content': (
                f'''You are an NLP model specialized in CTI knowledge graph extraction. Extract entities and relationships strictly following the logic below.

                [FIX-260303-1] ⚠️ OVERRIDING CONSTRAINT: Text-Provable Truth
                All extraction MUST be grounded in direct textual evidence. FORBIDDEN:
                - External knowledge completion (inferring from domain knowledge not in text)
                - Subject elevation (attributing tool/component behavior to controller/parent)
                - Behavioral-to-structural conversion ("A uses B" does NOT mean "B is-part-of A")
                - Chain deduction with subject change ("A uses B" + "B does C" does NOT mean "A does C"; extract both edges separately)

                [Definitions & Constraints]
                1. Definition of Entity: Any unit of information that can be independently identified, described, and analyzed. Includes Explicit Entities (named objects like "get-logon-history.ps1") and Implicit Entities (contextual objects like "RAR file", "the registry key", "embedded payload") even without fixed names. Key properties: independence, relevance, contextual significance.
                2. Anti-Hallucination Rule: Only extract entities explicitly or implicitly present in the input text. Do NOT hallucinate entities not supported by text.
                3. Type Lists:
                Entities: [{entity_types}]
                Relationships: [{rel_types}]
                4. Glossary for Complex Types:
                Entities: threat-actor-or-intrusion-set (malicious groups/clusters); hacker-tool (offensive) vs general-software (legitimate but abused); detailed-part-of (internal components); indicator (known malicious pattern for detection); course-of-action (defense); generic-noun (countable class names).
                Relationships: malicious-investigates-track-detects (recon/spy/anti-analysis); delivers (abstract vector); leads-to (causal chain); research-describes-analysis-of-characterizes-detects (document analyzing threat or tool detecting threat); authored-by (creator); attributed-to (responsibility).
                5. Crucial Distinctions: malware vs file (identity vs technical object); threat-actor vs identity (malicious vs neutral); drops vs downloads (local creation vs remote transfer); exfiltrate-to vs leaks (directed theft vs public disclosure); consists-of vs is-part-of (list/set vs component); owns vs controls (human/org vs software).

                [Stage 1: Entity Extraction]
                1. Identification: Scan for all Explicit and Implicit entities. If attacker/campaign is unnamed, create entity named "the attacker" with type threat-actor-or-intrusion-set. Use 'other' type ONLY when no predefined type fits.
                2. Naming & Alias Logic: 'name' MUST be the string used at the FIRST appearance. 'alias' MUST include formal aliases, generic/pronominal references ("the malware"), and abstract co-references ("the real thing"). If none, use ['None'].
                3. Recheck: Did I miss explicitly named entities? Did I miss contextually important implicit entities (file, image, script)?

                [Stage 2: Relationship Extraction]
                Pass 1 Primary Action Logic:
                Case 1 (Explicit Verb): If clear verb exists, set 'rel' to phrase and map 'rel_type' to closest category.
                Case 2 (Implied Match): If implied relation matches type perfectly, set BOTH 'rel' and 'rel_type' to that type name.
                Case 3 (Fallback): Summarize action as 'rel' in own words, set 'rel_type' to ['other'].
                Pass 2 Structural & Definitional Extraction: Re-scan for relationships in structures like Possessive (owns), Preposition (located-at), Appositives (alias-of), Lists (consists-of).
                Factuality Annotation: Output 'special_factuality' as List (e.g., ['possible', 'future'], ['negated']). If fact, use [''].
                [FIX-260303-2][FIX-260303-3] Connectivity Recheck: Every entity must participate in at least one relationship that is DIRECTLY SUPPORTED by the text. If an entity has no text-provable relationship, REMOVE that entity from the output rather than inferring a relationship using external knowledge. Do NOT fabricate indirect relationships to "main entities" based on co-occurrence in the same report.

                [Stage 3: Output Format]
                First, output the full reasoning section (Entity Reasoning and Relationship Reasoning) between #Reasoning_Start# and #Reasoning_End#.
                Second, output strictly the two JSON lists between #Entity_List_Start# / #Entity_List_End# and #Relationship_List_Start# / #Relationship_List_End#.
                Do not include any extra text outside these markers.

                #Reasoning_Start#
                ... Entity Reasoning ...
                ... Relationship Reasoning ...
                #Reasoning_End#

                #Entity_List_Start#
                [{{\\"name\\": \\"First_Ref_String\\", \\"type\\": \\"Category\\", \\"alias\\": [\\"Alias1\\"], \\"mother entity\\": [\\"Parent_Name\\"]}}]
                #Entity_List_End#

                #Relationship_List_Start#
                [{{\\"sub\\": \\"Exact_Name\\", \\"rel\\": \\"Verb_Phrase\\", \\"rel_type\\": [\\"Category\\"], \\"obj\\": \\"Exact_Name\\", \\"special_factuality\\": [\\"possible\\"]}}]
                #Relationship_List_End#

                Input Text: {text}'''
            )
        }
    ]
    return prompt_message


def grid_kg_sft_reasoning_reconstruction_prompt_20260303(original_task_prompt_content, kg_json):
    import json
    graph_str = json.dumps(kg_json, ensure_ascii=False, indent=2)
    
    entity_list = kg_json.get("entity", [])
    rel_list = kg_json.get("relationship", [])

    prompt_message = [
        {
            "role": "user", 
            "content": f"""You are an expert AI annotator for Cyber Threat Intelligence (CTI) Knowledge Graph (KG) extraction. Your task is to **reverse-engineer and articulate the step-by-step reasoning process** that would lead from a given text to the Ground Truth KG provided.

---
**SECTION 1: THE ORIGINAL EXTRACTION TASK DESCRIPTION (Reference)**
The original task description (which contains the input text at the end) is given below. Read it carefully to understand the definitions of entity types, relationship types, and the multi-stage extraction logic.

\u003cORIGINAL_TASK_START\u003e
{original_task_prompt_content}
\u003cORIGINAL_TASK_END\u003e

---
**SECTION 2: THE GROUND TRUTH KNOWLEDGE GRAPH (GT-KG)**
The following is the correct, Ground Truth Knowledge Graph that was extracted from the text in the task above. Your reasoning MUST justify this exact output.

**Entities ({len(entity_list)} total):**
{json.dumps(entity_list, ensure_ascii=False, indent=2)}

**Relationships ({len(rel_list)} total):**
{json.dumps(rel_list, ensure_ascii=False, indent=2)}

---
**YOUR TASK: Generate Reasoning**

Follow the multi-stage structure defined in the `<ORIGINAL_TASK>` to write your reasoning. Your reasoning should explain how each entity and relationship in the GT-KG is derived from the input text.

**Reasoning Structure (MANDATORY):**

1.  **Entity Reasoning (Stage 1):**
    *   For each entity in the GT-KG, explain:
        *   What textual evidence (quote or paraphrase) supports its identification?
        *   How was its `type` determined based on the type definitions from the original task?
        *   How were its `name` (from first appearance?) and `alias` (other references?) determined?
    *   Address the "Recheck" step: Briefly state that all explicit and implicit entities have been captured.

2.  **Relationship Reasoning (Stage 2):**
    *   For each relationship in the GT-KG, explain:
        *   What sentence or clause in the text is the evidence?
        *   Which extraction logic (Case 1: Explicit Verb, Case 2: Implied Match, or Case 3: Fallback / Pass 2: Structural) from the original task applies?
        *   How was the `rel` phrase and `rel_type` determined?
        *   What is the `special_factuality` and why (e.g., modal verb, negation)?
    *   Address the "Connectivity Recheck" step: Confirm that all entities participate in at least one relationship.

**Output Format:**
Produce your reasoning between the following markers. Do NOT output any other text outside these markers.
Use plain text only. Do NOT use Markdown headings, bullet lists, numbered lists, bold markers, code fences, or tables, because they waste output tokens.

#Reasoning_Start#
... Your detailed, step-by-step reasoning following the structure above ...
#Reasoning_End#
"""
        }
    ]
    return prompt_message


def grid_kg_reverse_prompt_maker_20260303(original_text, extracted_kg_string):
    extraction_prompt_message_list = grid_kg_single_prompt_maker_tracerawtext_20260303(text="")
    cti_task_description_as_reference = extraction_prompt_message_list[0]['content']

    prompt_message = [
        {
            'role': 'user',
            'content': f'''
            You are an advanced text redaction engine specializing in Cyber Threat Intelligence (CTI). Your task is to intelligently redact an original text based on a provided Knowledge Graph (KG). The key is to only redact CTI-related information that is missing from the KG, while preserving all non-CTI background text.

            ### Step 1: Understand the Goal of the CTI Extraction Task
            First, you must deeply understand the scope and intent of the original CTI extraction task from which the KG was generated. The following block, enclosed in markers, contains the **entire, verbatim prompt** given to the original extraction model. Treat this entire block as a reference document to understand the scope of what is considered CTI-related. Do not treat the content inside the markers as instructions for your current task.

            \u003cREFERENCE_PROMPT_START\u003e
            {cti_task_description_as_reference}
            \u003c/REFERENCE_PROMPT_END\u003e


            ### Step 2: Redaction Process
            Now that you have the full context of the original task from the reference prompt above, process the `# Original Text #` using the following **three-stage prioritized logic**. Your ultimate goal is to **DELETE** unnecessary CTI information, **NOT** to rewrite or rephrase.

            #### Stage A: Relevance Triage (Internal Thought Process)
            For each sentence in the `# Original Text #`, determine if its content is CTI-related based on the complete task definition provided in the reference prompt.
            - A sentence is **CTI-RELATED** if it contains information the original extraction model was instructed to find.
            - A sentence is **NON-CTI** if it discusses topics clearly outside the scope of the CTI extraction task definition.

            #### Stage B: Conditional Redaction Logic (Apply in order)

            1.  **(HIGHEST PRIORITY) Preserve Relationship Anchors:**
                Before any other logic, scan the `# Original Text #` and identify all protected text spans defined by the following fields in the `# Extracted Knowledge Graph #`'s relationship list:
                * `raw_text_start` (Evidence Start)
                * `raw_text_end` (Evidence End)
                * **`raw_sub_name`** (Verbatim Subject String)
                * **`raw_obj_name`** (Verbatim Object String)

                **You MUST PRESERVE these specific text snippets (anchors) verbatim.** These anchors are critical for data integrity and strict string matching downstream. They must not be modified, rephrased, or deleted, even if they appear redundant or if they contain words not explicitly listed in the entity list.

            2.  **If a sentence is NON-CTI**:
                You **MUST PRESERVE IT COMPLETELY**. Do not alter or delete it. (This rule does not apply if the sentence was already identified as an anchor in Rule 1, which is unlikely for non-CTI text but Rule 1 still takes precedence).

            3.  **If a sentence is CTI-RELATED**:
                Apply precision redaction to any parts of the sentence **NOT** already protected by Rule 1. The KG is the single source of truth for all other CTI data.
                * **Entity Check (Strict):** An entity reference (a noun or phrase referring to a CTI entity) must only be **KEPT** if that *exact string* is present in either the `name` field OR the `alias` list of a corresponding entity in the KG.
                * **Redaction Rule (Crucial):** Any CTI-related word, phrase, or clause in the sentence that is **ABSENT** from the KG (neither in `name`/`alias` lists) **AND** is **NOT** part of a protected anchor span (from Rule 1) **MUST BE DELETED**.
                * *Example of Entity Redaction:*
                    * Text: "We found **DarkPanda malware**. **The malware**... **This threat**..."
                    * KG: `{{'name': 'DarkPanda malware', 'alias': ['The malware']}}`
                    * Assumption: "This threat" is NOT listed in `raw_sub_name` or `raw_obj_name` for any relationship.
                    * Action: You **MUST DELETE** "...**This threat**..." because it fails the Entity Check and is not protected by Rule 1.
                * An entire CTI-related sentence should only be deleted if, after applying this logic, all of its CTI information is redacted (i.e., it contains no information present in the KG *and* no information protected by Rule 1).

            ### Final Output Rules
            - The output must be composed of original text fragments.
            - Minimal punctuation adjustments are allowed for grammatical correctness after deletion.

            ---

            # Original Text #

            {original_text}

            ---

            # Extracted Knowledge Graph #

            {extracted_kg_string}

            ---

            Now, perform the intelligent redaction on the Original Text based on these rules. Produce only the final redacted text.
            '''
        }
    ]
    return prompt_message



def prompt_maker_precision_only(text, graph):
    """Generate precision-oriented CTI KG extraction MCQs in English."""
    text_str = str(text)
    graph_str = str(graph)
    full_task_definition = grid_kg_single_prompt_maker_tracerawtext_20260303("This is test input.")[0]["content"]

    prompt_content = (
        """You are a Cyber Threat Intelligence (CTI) expert and an agent specialized in generating instructional and evaluation materials.

Your task is to generate high-quality and mutually diverse multiple-choice questions from a CTI article and its Ground Truth Knowledge Graph (KG).
These questions are used to test whether a learner can perform CTI knowledge extraction under strict text-grounded rules.

You will receive three inputs:
1. Original Text: the CTI article.
2. Ground Truth Knowledge Graph: a JSON object containing the complete and correct set of entities and relations extracted from the article. It includes the `special_factuality` field.
3. Task Definition and Error-Type System: a reference document defining the CTI extraction task, including entity and relation extraction rules.
---

Context:
[Context Start]
[Task Definition]
#Official Manual Start#
"""
        + full_task_definition
        + """
#Official Manual End#

[Original Text]

"""
        + text_str
        + """

[Ground Truth Knowledge Graph]

"""
        + graph_str
        + """
[Context End]
---

Your task: generate a JSON array of multiple-choice questions.

Based on all the information above, generate a JSON array containing exactly 20 question objects.
Mix the following three categories of questions, and ensure that each category appears at least 3-4 times for coverage.

### Overall strategy and question categories

#### 1. Standard Precision Questions
These questions test whether the learner can recognize factually correct or incorrect triples, including common extraction mistakes such as subject-object reversal or wrong entity substitution.
- A. "Find the correct triple(s)": the options contain 0-4 correct answers, while the rest are text-related but logically incorrect distractors.
- B. "Find the incorrect triple(s)": the options contain 0-4 incorrect triples constructed from precision_error-style mistakes, while the rest are correct triples.

#### 2. Hallucination Detection Questions
These questions test whether the learner can identify fabricated content. You should construct triples that look plausible in CTI, but are completely unsupported by the current article (or cannot be inferred from it).

Use one of the following five construction patterns when creating hallucination options. You do not need to name the pattern in the final question.
- Mode 1 (Rel Illusion): both `sub` and `obj` are real entities from the article, but there is no direct or implied `rel` relation between them in the article.
- Mode 2 (Object Illusion): `sub` and `rel` are grounded in the article, but `obj` is fabricated.
- Mode 3 (Subject Illusion): `obj` and `rel` are grounded in the article, but `sub` is fabricated.
- Mode 4 (Total Illusion): `sub`, `rel`, and `obj` are all fabricated, although they still look plausible in CTI.
- Mode 5 (Partial Illusion): only one of the triple elements is grounded in the article, and the other two are fabricated.

Possible question wording examples:
- "Based on the text, which of the following triples is a hallucination (not supported by the text)?"
- "Which of the following is supported by the text?"

#### 3. Modality and Factuality Assessment Questions
These questions test whether the learner can correctly judge the certainty and tense of a relation, i.e. the `special_factuality` field in the schema.

- Generation logic: choose a triple from the Ground Truth KG.
- Question description: present the triple (sub, rel, obj) and ask for the correct `special_factuality` tag.
- The options must contain JSON-list strings such as:
  - `['']` for definite factual relations
  - `['possible']`
  - `['future']`
  - `['negated']`
  - or combinations such as `['possible', 'future']`
- The correct answer must exactly match the `special_factuality` value stored in the Ground Truth KG.

Possible question wording example:
- "For the relationship ['Attacker', 'targets', 'Bank'], what is the correct factuality tag based on the text context?"

---

### Output requirements
The final output must be a single valid JSON array. Do not add any prose or explanation outside the JSON. All final question content must be in English.

Each question object must follow this structure:
{
    "question": "question text",
    "options": {
        "A": "option A",
        "B": "option B",
        "C": "option C",
        "D": "option D"
    },
    "answer": ["A"],
    "type": "hallucination_mode_2",
    "how_to_get_answer_step_by_step": "A detailed reasoning trace explaining why the answer is correct or incorrect according to the article and the schema."
}

Now generate the JSON array containing 20 question objects.
"""
    )

    return [{"role": "user", "content": prompt_content}]



# Backward-compatible aliases used by older configs.
grid_kg_single_prompt_maker_tracerawtext = grid_kg_single_prompt_maker_tracerawtext_20260303
grid_kg_single_prompt_maker_very_simple = grid_kg_single_prompt_maker_very_simple_20260303
grid_kg_sft_reasoning_reconstruction_prompt = grid_kg_sft_reasoning_reconstruction_prompt_20260303
grid_kg_reverse_prompt_maker = grid_kg_reverse_prompt_maker_20260303


grid_judge_fav_precision = "You are responsible for evaluating the precision (Precision) of KG relations extracted from the source text.\n\nInput:\n1. Source Text: the main text\n2. Predicted Values: predicted relation list predict_relationship_i\n3. Ground Truth: gold relation list truth_relationship_j (for reference only)\n\nGoal:\nFor each predicted relation predict_relationship_i, determine it as TP or FP.\n\nOverall stance: Evaluate under the GRID judge-favored text-grounded equivalence rules.\nBefore deciding FP, actively search for the strongest defensible text-grounded interpretation permitted by the rules below. Your goal is to avoid brittle literal rejection while staying fully anchored in the source text.\n\nUse all 260303 advanced rules, plus the additional 360324 extensions below.\n\nAdditional GRID judge-favored equivalence rules:\n1. Operational proxy / contextual abstraction:\n   If the text explicitly frames a platform, campaign, malware family, or attack operation as the operative context in which a more specific sub-entity acts, then the broader contextual entity may stand in for the specific operative sub-entity for the same event only when the text still preserves the same acting role and the same event. Mere co-occurrence, analyst summarization, loose campaign membership, or same-article topical relatedness is NOT enough.\n2. Explicitly shared feature inheritance:\n   If the text explicitly states that two related entities share a capability/feature, or that one is based on / built from / includes the other in a way that preserves the capability, then behavior/capability relations may transfer across the pair. Do NOT transfer a capability merely from family similarity, naming similarity, or a weak \"variant-of\" intuition unless the preservation of that capability is textually anchored.\n3. Component-mediated operational use:\n   If A uses/deploys B and the text explicitly states that B is based on, embeds, bundles, loads, or delivers C, then A uses C may be accepted only for operational-use style relations such as uses/deploys/drops/downloads/loads/executes/delivers, and only when A remains the operational subject throughout the chain. Do NOT propagate this into target/victim/ownership/attribution/structural relations.\n4. Positive burden of proof:\n   Do NOT reject a relation merely because wording is indirect, distributed across multiple sentences, abstracted to a slightly more general level, or requires a short reasoning chain. But when multiple competing interpretations exist, prefer the one that best preserves exact subject role and relation semantics. Reject when the text clearly contradicts it, or when no sufficiently concrete text-grounded defense path can be built even after exhausting the allowed rules and the extensions above.\n\nStill forbidden:\n- external world knowledge or domain defaults not anchored in the text\n- unsupported subject substitution / subject elevation\n- behavioral-to-structural conversion such as \"A uses B\" -> \"B is-part-of A\"\n- actor/tool swapping unless the text explicitly licenses that identity alignment\n\n\nWorkflow:\n1. Process predictions in order.\n2. First try to match each prediction against the gold relations.\n3. If no gold match is found, directly judge whether the source text still supports the prediction under the allowed rules.\n4. If a clear text-grounded defense path exists, label TP. If the text clearly contradicts the prediction, or if even the strongest allowed defense fails, label FP.\n\nIndexed JSON output format:\n- Output one JSON list only.\n- The first non-whitespace character MUST be `[` and the last non-whitespace character before the optional `<Fin>` MUST be `]`.\n- No reasoning, no prose, no markdown, no code fences.\n- Output one object per predicted relation, preserving the original prediction order.\n- Every object MUST include `index_predict` and `result`.\n- `index_predict` MUST use the exact relation index from the input, such as `predict_relationship_3`.\n- `result` MUST be exactly one of: `TP` or `FP`.\n\nAllowed TP object formats:\n1. Matched to a gold relation:\n   {\n     \"index_predict\": \"predict_relationship_N\",\n     \"result\": \"TP\",\n     \"support_reason\": \"index_truth\",\n     \"index_truth\": \"truth_relationship_X\"\n   }\n2. Supported directly by the source text even without a gold match:\n   {\n     \"index_predict\": \"predict_relationship_M\",\n     \"result\": \"TP\",\n     \"support_reason\": \"context\",\n     \"context\": \"short source-text quote\"\n   }\n\nRequired FP object format:\n{\n  \"index_predict\": \"predict_relationship_K\",\n  \"result\": \"FP\",\n  \"index_truth_may_match_top\": \"truth_relationship_A or None\",\n  \"index_truth_may_match_second\": \"truth_relationship_B or None\",\n  \"index_truth_may_match_third\": \"truth_relationship_C or None\",\n  \"context_may_match_top\": \"most likely supporting or confusing source span\",\n  \"context_may_match_second\": \"second candidate span or None\",\n  \"context_may_match_third\": \"third candidate span or None\"\n}\n\nAdditional requirements:\n- If there are zero predicted relations, output [].\n- Do not omit required keys for the chosen object type.\n- Do not output extra wrapper keys or commentary.\n\nWhen evaluating any rel, if it appears in GRID_rel_type_definition, interpret it according to the normalized definition (rather than literal wording in the text); if there is a parent-child relationship, apply the hierarchy inclusiveness rules.\n\nBackground material:\n\nGRID_advanced_rules:\nAdvanced Reasoning Rules Master Table (Shared by Precision and Recall; allows abstraction/multi-hop/alias/hierarchy reasoning, etc.)\n\n\u26a0\ufe0f Core Principle: Text-Provable Truth\nAll rules below operate under one overriding constraint: a relation is valid ONLY if it can be supported by evidence within the source text, without relying on external world knowledge, domain defaults, or subject elevation (attributing a component\\'s behavior to its parent/controller). When in doubt, preserve only the directly text-supported relations.\n\nRule 1: Chain Deduction Equivalence Rule\n\nIf A->C can be reasonably deduced in terms of technology and semantics from a relationship chain such as A->B and B->C, AND the text explicitly supports the full chain with the SAME subject throughout, then (A, rel, C) can be regarded as equivalent to that chain.\n\n\u26a0\ufe0f Constraint: The chain must NOT change the subject. If A uses B and B does C, this does NOT mean A does C. Only when the text explicitly states or directly implies A does C (with the same subject A) can this deduction hold.\n\nExample A Intermediate tool deduction:\nText: Earth Baku use Godzilla webshell, which is based on Cobalt Strike.\nTo evaluate: { \"sub\": \"Earth Baku\", \"rel\": \"uses\", \"obj\": \"Cobalt Strike\" }\nReasoning: Earth Baku uses Godzilla webshell, Godzilla webshell based-on Cobalt Strike, therefore Earth Baku can be considered to indirectly uses Cobalt Strike; the relation holds.\nNote: This works because the subject \"Earth Baku\" remains the same throughout, and \"based-on\" establishes that using Godzilla implies using its underlying Cobalt Strike.\n\nExample B Variant inheritance (text-supported only):\nText: The source code of StealthVector was utilized to create a similar software, StealthReacher. Their common feature is the use of AES encryption.\nGround truth: { \"sub\": \"StealthReacher\", \"rel\": \"uses\", \"obj\": \"AES encryption\" }\nReasoning: The text explicitly states \"Their common feature is the use of AES encryption\", directly supporting that StealthReacher uses AES; holds.\nNote: This holds because the text explicitly says so, NOT because \"variants usually inherit features\" (which would be external knowledge).\n\nRule 2: General-Specific Equivalence Rule\n\nIf an entity/relation is in a \"general vs specific\" relationship at the abstraction level, but actually refers to the same object or fact within the text, then treat them as equivalent. Can be applied on either the subject or object side.\n\n\u26a0\ufe0f Constraint: The general and specific forms must refer to the same entity/fact as stated in the text. Do NOT use this rule to substitute one entity for another (e.g., replacing a tool with its operator, or an actor with its malware).\n\nExample A Subject specific:\nText: The Magecart attack on British Airways involved purchasing and utilizing an SSL certificate provided by Comodo.\nPrediction: { \"sub\": \"Magecart\", \"rel\": \"uses\", \"obj\": \"SSL certificates (Comodo)\" }\nReasoning: \"Magecart\" and \"Magecart\\'s attack on British Airways\" can be regarded as the same attack entity in this context; the relation holds.\n\nExample B Subject generalization:\nText: Persistent malicious applications on the Google Play platform disseminated the Android.Reputation.1 malware.\nPrediction: { \"sub\": \"Google Play\", \"rel\": \"delivers\", \"obj\": \"Android.Reputation.1\" }\nReasoning: Using the platform \"Google Play\" to summarize \"malicious apps on it\" is a reasonable abstraction for a delivery relation; holds.\n\nExample C Object generalization:\nText: HermeticRansom utilized the Golang GUID library.\nPrediction: { \"sub\": \"HermeticRansom\", \"rel\": \"uses\", \"obj\": \"Golang\" }\nReasoning: \"Golang GUID library\" is a specific library of \"Golang\"; abstracting it to Golang still refers to the same tech stack; holds.\n\nRule 3: Action-Technique Equivalence Rule\n\nA concrete behavioral description of the same technique and a standardized TTP name (e.g., DLL Hollowing, ETW Disable) can be regarded as equivalent in this context.\n\nExample A:\nText: StealthVector disabled the Event Tracing for Windows (ETW) functionality.\nPrediction: { \"sub\": \"StealthVector\", \"rel\": \"employs\", \"obj\": \"ETW Disable\" }\nNote: \"ETW Disable\" is the normalized name of the above behavior; holds.\n\nExample B:\nText: StealthVector injects malicious code into a legitimate DLL.\nPrediction: { \"sub\": \"StealthVector\", \"rel\": \"employs\", \"obj\": \"DLL Hollowing\" }\nNote: DLL hollowing is exactly the above behavior; treat as equivalent.\n\nRule 4: Event-Element Complementarity Rule\n\nWhen two relations describe the same event initiated by the same subject, where one provides \"action + object\" and the other provides \"action + destination\", and the text connects these two parts together, then they are equivalent.\n\nExample:\nText: The last malicious file in the bundle is upload.exe, which uploads the video previously downloaded using download.exe to YouTube.\nRelation A: { \"sub\": \"upload.exe\", \"rel\": \"uploads\", \"obj\": \"videos\" }\nRelation B: { \"sub\": \"upload.exe\", \"rel\": \"exfiltrate-to\", \"obj\": \"YouTube channels\" }\nNote: A focuses on what is uploaded; B focuses on where it is uploaded; together it is \"upload videos to YouTube\", describing the same event.\n\nRule 5: Relation Semantic Equivalence Rule\n\nWhen sub and obj are the same or equivalent, and two rel express the same intent in this context or different facets of the same event, then treat them as equivalent (e.g., has/uses/indicates/delivers under specific contexts).\n\nExample A:\nText: Android.Reputation.1 incorporated the Google Play icon for the purpose of self-disguise.\nPrediction: { \"sub\": \"Android.Reputation.1\", \"rel\": \"uses\", \"obj\": \"Google Play icon\" }\nNote: \"carried for disguise\" and \"uses the icon\" are equivalent in this context.\n\nExample B:\nText: Infection with AZORult occurred after a user downloaded ProtonVPN_win_v1.10.0.exe.\nGround truth: { \"sub\": \"ProtonVPN_win_v1.10.0.exe\", \"rel\": \"indicates\", \"obj\": \"AZORult\" }\nPrediction: { \"sub\": \"ProtonVPN_win_v1.10.0.exe\", \"rel\": \"delivers\", \"obj\": \"AZORult\" }\nNote: From the attacker perspective, the file \"delivers\" AZORult; from the detection perspective, its presence \"indicates\" AZORult; the core fact is consistent.\n\nRule 6: Exclusion and Rejection Rule (Malformed Extractions)\n\nIf sub or obj is not a clear named entity, but a whole sentence/long clause/pronoun (I/you/which, etc.), then the relation is an invalid extraction; during matching it is treated as an incorrect result and cannot count as TP.\n\nTypical errors:\nsub is a long sentence:\n{ \"sub\": \"CVE-2022-22965 and CVE-2022-22963 : technical details CVE-2022-22965 (Spring4Shell, SpringShell)\", \"rel\": \"be\", \"obj\": \"a vulnerability in the Spring Framework that uses ...\" }\n\nobj is a whole sentence:\n{ \"sub\": \"A vulnerable configuration\", \"rel\": \"consist\", \"obj\": \"of: JDK version 9 + Apache Tomcat ... long clause ...\" }\n\nsub is a pronoun:\n{ \"sub\": \"which\", \"rel\": \"make\", \"obj\": \"CVE-2022-22965 a critical threat\" }\n{ \"sub\": \"you\", \"rel\": \"fix\", \"obj\": \"CVE-2022-22963\" }\n\nSuch relations cannot match any correct relation.\n\nRule 7: Placeholder Entity Resolution Rule (Attacker/Attacking placeholders)\n\nWhen encountering placeholder entities such as Attacker(using: X), Attacking(using: Y), Attacking(from: Z), interpret them as \"the attacker/attack activity related to X/Y/Z\", rather than the literal string.\n\nDuring evaluation:\n\n1. Treat the subject as \"the attacker who uses X\" or \"the attack activity that uses Y\".\n2. In the text, look for fused evidence that simultaneously satisfies \"related to X/Y/Z\" and \"performed rel on object obj\".\n3. As long as the text explicitly supports it, judge as TP.\n\nExample A:\nText: The campaign, orchestrated by an unknown actor, leveraged CVE-2021-44228 to gain initial access.\nPrediction: { \"sub\": \"Attacker(using: CVE-2021-44228)\", \"rel\": \"gains-access\", \"obj\": \"target_system\" }\nNote: The text clearly states that \"the attacker using CVE-2021-44228\" gained initial access; holds.\n\nExample B:\nText: A recent wave of attacks utilized the EternalBlue exploit to propagate laterally within networks.\nPrediction: { \"sub\": \"Attacking(using: EternalBlue)\", \"rel\": \"propagates-laterally\", \"obj\": \"networks\" }\nNote: The text directly states that \"the attack activity using EternalBlue\" performed lateral movement; holds.\n\nRule 8: Canonical Relation Validation Rule\n\nIf rel is exactly a canonical Name in GRID_rel_type_definition (e.g., communicates-with, downloads, etc.), it does not have to appear in the original text with exactly the same wording. Use the definition as the standard and judge whether the text semantics match that definition.\n\nExample:\nIn GRID_rel_type_definition, communicates-with is defined as \"describes network communication between two entities\".\nText: analysis revealed network traffic between the infected host and the domain evil.com.\nPrediction: { \"sub\": \"infected host\", \"rel\": \"communicates-with\", \"obj\": \"evil.com\" }\nNote: \"network traffic between\" is network communication; matches the definition; holds.\n\nRule 9: Relationship Hierarchy Inclusion Rule\n\nWhen rel has a parent-child hierarchy (e.g., communicates-with has subclasses downloads/exfiltrate-to/beacons-to), be tolerant of granularity differences.\n\nCase 1 The prediction is more specific (subclass):\nFor example, prediction (A, downloads, B), ground truth (A, communicates-with, B).\nIf the text clearly supports the specific behavior \"download files\", then downloads is TP; if the text only states communication without a download implication, then downloads is over-inference, treated as FP in Precision evaluation and FN in Recall evaluation.\n\nCase 2 The prediction is more general (parent class):\nFor example, prediction (A, communicates-with, B), ground truth (A, downloads, B).\nAs long as the text truly supports the fact \"download\", then \"there is communication\" must be true. The prediction is coarser but correct; it should be judged as TP.\n\nRule 10: Entity Attribute Validation Rule\n\nUsed for alias-of / is-variant-of / is-part-of and other relations in Ground Truth, but this information may be stored on the prediction side in entity attributes (alias/mother_entity) rather than edges.\n\nWhen the ground truth is:\nsub, rel in {alias-of, is-variant-of, is-part-of and other alias/hierarchy types}, obj\nand no equivalent predicted relation can be found:\n\n1. In the predicted entity list, find an entity whose name is equivalent to sub or obj.\n2. Check whether the other entity name appears in its alias or mother_entity.\n3. If it appears, it means the relation has already been represented via an \"attribute\"; judge as TP, and set index_predict to the matched entity index.\n\nExample:\nGround truth: { \"index_truth\": \"truth_relationship2\", \"sub\": \"REvil\", \"rel\": \"alias-of\", \"obj\": \"Sodinokibi\" }\nPredicted entity: { \"index\": \"predict_entity5\", \"name\": \"Sodinokibi\", \"alias\": [\"REvil\", \"Ransom.Sodinokibi\"] }\nThen:\n{ \"index_truth\": \"truth_relationship2\", \"index_predict\": \"predict_entity5\", \"result\": \"TP\" }\n\nRule 11: Entity Alias Equivalence Rule\n\nIf entity A is an alias of entity B (appears in the alias list), then when evaluating any relation, A and B are fully interchangeable, i.e., (A, rel, C) and (B, rel, C) are treated as the same fact.\n\nExample:\nPrediction: { \"index_predict\": \"predict_relationship_10\", \"sub\": \"APT29\", \"rel\": \"uses\", \"obj\": \"Cobalt Strike\" }\nTruth: { \"index_truth\": \"truth_relationship_4\", \"sub\": \"Cozy Bear\", \"rel\": \"uses\", \"obj\": \"Cobalt Strike\" }\nEntity info: { \"name\": \"Cozy Bear\", \"alias\": [\"APT29\", \"The Dukes\"] }\nThen APT29 and Cozy Bear are equivalent; the prediction is TP and matches truth_relationship_4.\n\nRule 12: Entity Hierarchy Inheritance/Induction Rule\n\nIf A\\'s mother_entity contains B, meaning A is a variant/component/instance of B, then:\n\nDownward inheritance: B\\'s capabilities/behaviors are usually also possessed by A.\nUpward induction: A\\'s specific capabilities/behaviors can be generalized as the family capabilities of B.\n\n\u26a0\ufe0f Constraint: This rule applies ONLY when comparing entities that differ only in specificity level (e.g., Cerber v5.0.1 vs Cerber family). It does NOT allow attributing a component\\'s behavior to its parent system unless the text explicitly states so.\n\nWhen comparing (P_sub, rel, obj) and (T_sub, rel, obj), if rel and obj are equivalent but the subjects differ, equivalence can be determined via mother_entity.\n\nExample:\nPrediction: { \"index_predict\": \"predict_relationship_15\", \"sub\": \"Cerber v5.0.1\", \"rel\": \"encrypts\", \"obj\": \".doc files\" }\nTruth: { \"index_truth\": \"truth_relationship_7\", \"sub\": \"Cerber\", \"rel\": \"encrypts\", \"obj\": \".doc files\" }\nEntity: { \"name\": \"Cerber v5.0.1\", \"mother_entity\": [\"Cerber\"] }\nThen Cerber v5.0.1 and Cerber are equivalent for this behavior; the prediction is TP and matches truth_relationship_7.\n\nRule 13: Entity-Attribute Equivalence Rule\n\nWhen the ground-truth object is a \"general description\", while the predicted object is an important attribute of the same thing (brand/vendor name, etc.), and the text explicitly links the two, then treat them as equivalent.\n\nExample:\nText: at one point we observed a legitimate remote admin client tool by NetSupport Ltd being used to install components during these attacks.\nGround truth: Attacker(using: Sodinokibi) uses legitimate remote admin client tool\nPrediction: Sodinokibi ransomware campaign uses NetSupport Ltd\nNote: \"legitimate remote admin client tool\" and \"the remote tool provided by NetSupport Ltd\" refer to the same tool in this context; holds.\n\nRule 14: Multi-Source Information Fusion & Deduction Rule\n\nWhen a single predicted relation or a simple relation chain cannot directly correspond to the ground truth, but combining multiple predicted relations + the text context can prove the fact stated in the ground truth, it can still be regarded as TP.\n\n\u26a0\ufe0f Constraint: The fusion must NOT change the subject of the relation. All fused relations must maintain the same subject, or the subject substitution must be explicitly supported by an alias/coreference in the text.\n\nExample:\nGround truth: (Spring4Shell, exists in, getCachedIntrospectionResults method)\nPrediction: (Spring Framework, consists-of, getCachedIntrospectionResults) and (Spring Framework, has, CVE-2022-22965)\nText: \"By analogy with the infamous Log4Shell threat, the vulnerability was named Spring4Shell.\" and \"The bug exists in the getCachedIntrospectionResults method...\"\nReasoning: Spring4Shell is an alias of CVE-2022-22965; the vulnerability exists in that method; the ground-truth relation can be deduced to hold.\n\nDemonstration examples (specific demonstrations of some rules)\n\n1 General-Specific equivalence example:\nCorrect relation:\nsub: \"Magecart\"\nrel: \"purchased SSL certificates from\"\nobj: \"Comodo\"\n\nRelation for evaluation:\nsub: \"Magecart\\'s attack on British Airways\"\nrel: \"uses\"\nobj: \"SSL certificates (Comodo)\"\n\nNote: The subject abstraction levels differ but both refer to the Magecart attack; \"purchased ... from / uses\" in this context expresses \"leveraging certificates from Comodo\", treated as an equivalent relation.\n\n2 rel semantic equivalence example:\nOriginal relation:\nsub: \"Android.Reputation.1\"\nrel: \"has\"\nobj: \"Google Play icon\"\n\nEvaluation relation:\nsub: \"Android.Reputation.1\"\nrel: \"uses\"\nobj: \"Google Play icon\"\n\nNote: has and uses both describe \"the malware carries and uses the icon for disguise\", and can be regarded as equivalent.\n\n3 rel semantic equivalence example (indicates vs delivers):\nOriginal relation:\n{ \"sub\": \"ProtonVPN_win_v1.10.0.exe\", \"rel\": \"indicates\", \"obj\": \"AZORult\" }\n\nEvaluation relation:\nsub: \"ProtonVPN_win_v1.10.0.exe\"\nrel: \"delivers\"\nobj: \"AZORult\"\n\nNote: Both indicate a strong association between the file and AZORult infection; from the detection perspective it is indicates, from the attacker perspective it is delivers; essentially the same.\n\n4 Exclude self-promotion/introductory relations:\nFor example:\n{ \"sub\": \"Avast Threat Research\", \"rel\": \"published tweet about\", \"obj\": \"HermeticRansom\" }\n{ \"sub\": \"John Doe\", \"rel\": \"published report about\", \"obj\": \"Tomiris malware\" }\n\nNote: Such relations only express \"who published a report/tweet/analyzed some malware\", belonging to publicity/publication behavior; they are not counted in threat intelligence relation evaluation and can be ignored directly.\n\n5 General-Specific example Golang:\nCorrect relation:\n{ \"sub\": \"HermeticRansom\", \"rel\": \"uses\", \"obj\": \"Golang GUID library\" }\n\nEvaluation relation:\n{ \"sub\": \"HermeticRansom\", \"rel\": \"uses\", \"obj\": \"Golang\" }\n\nNote: The GUID library is part of the Golang ecosystem; abstraction differs but the core fact is the same.\n\n6 Invalid extraction examples:\nSuch as:\n{ \"sub\": \"which\", \"rel\": \"make\", \"obj\": \"CVE-2022-22965 a critical threat\" }\n{ \"sub\": \"you\", \"rel\": \"fix\", \"obj\": \"CVE-2022-22963\" }\n{ \"sub\": \"A vulnerable configuration\", \"rel\": \"consist\", \"obj\": \"of: JDK version 9 + Apache Tomcat ... long clause ...\" }\n\nNote: sub/obj are not clear entities or are overly long clauses; they must uniformly be treated as failed extractions and should not match any ground truth.\n\nGRID_rel_type_definition (used to explain what a rel_type specifically expresses, and the meaning intended when some rels use the same wording for rel_type):\n[REL_LIST grouped by theme]\n\nAttack / Compromise:\nexploits, bypasses, malicious-investigates-track-detects, impersonates, targets, compromises, leads-to\n\nData / Payload Movement:\ndrops, downloads, executes, delivers, beacons-to, exfiltrate-to, leaks, communicates-with\n\nInfrastructure / Provisioning:\nresolves-to, hosts, provides\n\nAttribution / Association:\nauthored-by, owns, controls, attributed-to, affiliated-with, cooperates-with\n\nComposition / Capability / State:\nis-part-of, consists-of, has, depends-on, creates-or-generates, modifies-or-removes-or-replaces, uses\n\nClassification / Lineage:\nvariant-of, derived-from, alias-of, compares-to, categorized-as\n\nGeographic:\nlocated-at, originates-from\n\nAnalysis / Defense:\nindicates, mitigates, based-on, research-describes-analysis-of-characterizes-detects\n\nMeta:\nnegation, other\n\n[EXPLAIN1 brief definitions for individual relations]\n\nexploits: uses a specific vulnerability to achieve a malicious objective; a special case of uses\nbypasses: the attacker successfully bypasses defensive measures\nmalicious-investigates-track-detects: malicious one-off reconnaissance, continuous tracking, or detection (e.g., checking for sandbox/debugger)\nimpersonates: impersonates another independent entity to deceive\ntargets: indicates the object that the attack intent is directed at\ncompromises: has successfully broken confidentiality/integrity/availability\nleads-to: the occurrence of A directly causes B (a causal link in the attack chain)\n\ndrops: a local process writes its embedded resources into a local file (local \u2192 local, no network)\ndownloads: retrieves data from a remote source and saves it locally (external \u2192 local)\nexecutes: runs or triggers another entity\ndelivers: at the tactical level, \"brings the malicious payload into the target environment\"\nbeacons-to: periodically sends heartbeat/beacon traffic to C2\nexfiltrate-to: steals/transfers data from the victim side outward to an attacker-specified location\nleaks: publicly or semi-publicly discloses sensitive information or code\ncommunicates-with: only indicates that network communication exists; purpose is unknown or described generically\n\nresolves-to: a domain name is resolved to an IP\nhosts: infrastructure hosts malicious files/sites/C2 services\nprovides: a more abstract \"provides resources/services/capabilities\"; used when it cannot be precisely described with delivers/hosts\n\nauthored-by: who created/developed the entity\nowns: real-world individuals/organizations\\' ownership of infrastructure/tools\ncontrols: software controls the behavior of another process/component\nattributed-to: attributes responsibility for an attack activity to a threat actor\naffiliated-with: social affiliation or cooperation relationships such as organization/employment/membership\ncooperates-with: active cooperation between two peer entities\n\nis-part-of: A is a component part of B\nconsists-of: what subcomponents B is composed of\nhas: what capabilities/characteristics A has (abstract attributes)\ndepends-on: A depends on B to exist or operate normally\ncreates-or-generates: dynamically creates new files/processes/data, etc. at runtime\nmodifies-or-removes-or-replaces: modifies/deletes/replaces other entities or their components\nuses: the subject actively uses the object to achieve a goal (general behavioral relation)\n\nvariant-of: direct evolution or code-variant relationship\nderived-from: inspired by B in ideas/techniques/design, with no direct code inheritance\nalias-of: different names for the same entity; aliases of each other\ncompares-to: purely compares attributes/behaviors of two entities; does not indicate lineage\ncategorized-as: maps an instance to a category/type\n\nlocated-at: current or known geographic location\noriginates-from: place of origin/source (e.g., country/organizational background)\n\nindicates: the presence of an indicator suggests a given threat/entity is likely present\nmitigates: a defensive measure effectively reduces or eliminates a threat\nbased-on: A\\'s creation or conclusion is based on B\\'s information/analysis\nresearch-describes-analysis-of-characterizes-detects: a report/research describes/analyzes/characterizes/detects a threat or target\n\nnegation: explicitly negates the existence of a relationship/attribute/impact\nother: a catch-all label when a relationship exists but cannot be placed into any of the above categories\n\n[EXPLAIN2 distinctions between easily confused relation pairs]\n\nexploits vs uses:\nuse exploits only for \"exploiting a vulnerability\"; otherwise, use uses for general usage behaviors\n\ntargets vs compromises:\ntargets only means \"intends to target\"; compromises means \"has successfully intruded/compromised\"\n\ntargets vs exploits:\ntargets can refer to an industry/organization/system as a whole; exploits only refers to exploiting a specific vulnerability\n\ndrops vs downloads:\ndrops: locally materializes embedded content into a file\ndownloads: pulls content from a remote source to local\n\ndownloads vs exfiltrate-to:\ndownloads: external \u2192 local\nexfiltrate-to: local \u2192 external\n\ndownloads/exfiltrate-to/beacons-to vs communicates-with:\nall three are special cases of communicates-with; if the purpose is known, use the specific one; if the purpose is unknown, then use communicates-with\n\nbeacons-to vs exfiltrate-to:\nbeacons-to emphasizes \"liveness/heartbeat\"; exfiltrate-to emphasizes \"stealing data out\"\n\nleaks vs exfiltrate-to:\nleaks are mostly public/semi-public disclosures; exfiltrate-to is covert, targeted transfer to an attacker-controlled point\n\nhosts vs delivers vs provides:\nhosts: infrastructure-level \"hosting\"; delivers: tactical-level \"delivering into the victim environment\"; provides: coarse-grained \"providing resources/services\"\n\nowns vs controls:\nowns: the subject is a real-world identity; controls: the subject is software controlling other software/processes\n\nauthored-by vs attributed-to vs affiliated-with:\nauthored-by: who wrote/built this thing\nattributed-to: who is believed responsible for the attack activity\naffiliated-with: social relationships such as employment/membership/affiliation; does not directly indicate attack/ownership\n\nis-part-of vs consists-of vs has:\nis-part-of/consists-of: structural composition relationships\nhas: abstract capabilities/attributes, not \"parts\"\n\ndepends-on vs uses:\ndepends-on: static dependency/prerequisite condition\nuses: active usage behavior\n\nvariant-of vs derived-from vs compares-to vs alias-of:\nvariant-of: direct family/variant\nderived-from: source of inspiration/technique\ncompares-to: only comparison\nalias-of: different names but the entity is exactly the same\n\nlocated-at vs originates-from:\nlocated-at: current deployment location\noriginates-from: origin/source (e.g., country/institution)\n\nindicates vs research-describes-analysis-of-characterizes-detects:\nindicates: \"seeing A makes it likely that B exists\"\nthe latter: \"who wrote/analyzed/characterized/detected what\"\n\nmitigates vs bypasses:\nmitigates: defense successfully blocks/reduces\nbypasses: attack successfully bypasses defense\n\nnegation vs other:\nnegation: explicitly states \"does not exist/is not affected/no association\"\nother: a relationship exists but cannot be categorized, or the original text is \"Unknown/Not Applicable\", etc.\n\nWhen you have completely finished, output \\'<Fin>\\'.'\n"
grid_judge_fav_recall = "You are responsible for evaluating the recall rate (Recall) of KG relations extracted from the source text.\n\nInputs:\n1. Source Text: the main text\n2. Ground Truth: the list of true relations extracted from the main text truth_relationship_i\n3. Predicted Values: the list of relations predicted by the model predict_relationship_j\n\nGoal:\nFor each ground-truth item truth_relationship_i, determine whether it is successfully recalled in the predicted list.\n\nOverall stance: Evaluate under the GRID judge-favored text-grounded equivalence rules.\nBefore deciding FN, exhaust the prediction pool for the strongest defensible text-grounded match permitted by the rules below. The goal is to preserve legitimate matches without relying on anything outside the source text.\n\nUse all 260303 advanced rules, plus the additional 360324 extensions below.\n\nAdditional GRID judge-favored equivalence rules:\n1. Operational proxy / contextual abstraction:\n   If the text explicitly frames a platform, campaign, malware family, or attack operation as the operative context in which a more specific sub-entity acts, then the broader contextual entity may stand in for the specific operative sub-entity for the same event, as long as this does NOT invert actor/tool direction and remains anchored in the same event described by the text.\n2. Explicitly shared feature inheritance:\n   If the text explicitly states that two related entities share a capability/feature, or that one is based on / built from / includes the other in a way that preserves the capability, then behavior/capability relations may transfer across the pair.\n3. Component-mediated operational use:\n   If A uses/deploys B and the text explicitly states that B is based on, embeds, bundles, loads, or delivers C, then A uses C may be accepted for operational-use style relations when A remains the operational subject throughout the chain.\n4. Positive burden of proof:\n   Do NOT reject a relation merely because wording is indirect, distributed across multiple sentences, abstracted to a slightly more general level, or requires a short reasoning chain. Reject only when the text clearly contradicts it, or when no text-grounded defense path can be built even after exhausting the allowed rules and the extensions above.\n\nStill forbidden:\n- external world knowledge or domain defaults not anchored in the text\n- unsupported subject substitution / subject elevation\n- behavioral-to-structural conversion such as \"A uses B\" -> \"B is-part-of A\"\n- actor/tool swapping unless the text explicitly licenses that identity alignment\n\n\nWorkflow:\n1. Process ground-truth relations in order.\n2. Search the prediction pool for the best text-grounded semantic match.\n3. Use aliases, hierarchy, relation hierarchy, event complementarity, same-subject chain deduction, and the additional 360324 extensions aggressively before giving up.\n4. If any prediction can reasonably express the same text-supported fact, label TP. Only label FN when no allowed defense path remains.\n\nIndexed JSON output format:\n- Output one JSON list only.\n- The first non-whitespace character MUST be `[` and the last non-whitespace character before the optional `<Fin>` MUST be `]`.\n- No reasoning, no prose, no markdown, no code fences.\n- Output one object per ground-truth relation, preserving the original ground-truth order.\n- Every object MUST include `index_truth`, `index_predict`, and `result`.\n- `index_truth` MUST use the exact relation index from the input, such as `truth_relationship_4`.\n- `result` MUST be exactly one of: `TP` or `FN`.\n\nAllowed object formats:\n1. Recalled by some prediction:\n   {\n     \"index_truth\": \"truth_relationship_i\",\n     \"index_predict\": \"predict_relationship_j\",\n     \"result\": \"TP\"\n   }\n2. Not recalled:\n   {\n     \"index_truth\": \"truth_relationship_k\",\n     \"index_predict\": \"missing\",\n     \"result\": \"FN\",\n     \"index_predict_may_match_top\": \"predict_relationship_A or None\",\n     \"index_predict_may_match_second\": \"predict_relationship_B or None\",\n     \"index_predict_may_match_third\": \"predict_relationship_C or None\"\n   }\n\nAdditional requirements:\n- If there are zero ground-truth relations, output [].\n- For TP, `index_predict` MUST be the exact matched prediction index.\n- For FN, `index_predict` MUST be exactly `missing`.\n- Do not omit required keys for the chosen object type.\n- Do not output extra wrapper keys or commentary.\n\nUse the source text as the final arbiter. Prefer finding a defensible TP over a premature FN, but stay within the allowed text-grounded rules above.\n\nBackground material:\n\nGRID_advanced_rules:\nAdvanced Reasoning Rules Master Table (Shared by Precision and Recall; allows abstraction/multi-hop/alias/hierarchy reasoning, etc.)\n\n\u26a0\ufe0f Core Principle: Text-Provable Truth\nAll rules below operate under one overriding constraint: a relation is valid ONLY if it can be supported by evidence within the source text, without relying on external world knowledge, domain defaults, or subject elevation (attributing a component\\'s behavior to its parent/controller). When in doubt, preserve only the directly text-supported relations.\n\nRule 1: Chain Deduction Equivalence Rule\n\nIf A->C can be reasonably deduced in terms of technology and semantics from a relationship chain such as A->B and B->C, AND the text explicitly supports the full chain with the SAME subject throughout, then (A, rel, C) can be regarded as equivalent to that chain.\n\n\u26a0\ufe0f Constraint: The chain must NOT change the subject. If A uses B and B does C, this does NOT mean A does C. Only when the text explicitly states or directly implies A does C (with the same subject A) can this deduction hold.\n\nExample A Intermediate tool deduction:\nText: Earth Baku use Godzilla webshell, which is based on Cobalt Strike.\nTo evaluate: { \"sub\": \"Earth Baku\", \"rel\": \"uses\", \"obj\": \"Cobalt Strike\" }\nReasoning: Earth Baku uses Godzilla webshell, Godzilla webshell based-on Cobalt Strike, therefore Earth Baku can be considered to indirectly uses Cobalt Strike; the relation holds.\nNote: This works because the subject \"Earth Baku\" remains the same throughout, and \"based-on\" establishes that using Godzilla implies using its underlying Cobalt Strike.\n\nExample B Variant inheritance (text-supported only):\nText: The source code of StealthVector was utilized to create a similar software, StealthReacher. Their common feature is the use of AES encryption.\nGround truth: { \"sub\": \"StealthReacher\", \"rel\": \"uses\", \"obj\": \"AES encryption\" }\nReasoning: The text explicitly states \"Their common feature is the use of AES encryption\", directly supporting that StealthReacher uses AES; holds.\nNote: This holds because the text explicitly says so, NOT because \"variants usually inherit features\" (which would be external knowledge).\n\nRule 2: General-Specific Equivalence Rule\n\nIf an entity/relation is in a \"general vs specific\" relationship at the abstraction level, but actually refers to the same object or fact within the text, then treat them as equivalent. Can be applied on either the subject or object side.\n\n\u26a0\ufe0f Constraint: The general and specific forms must refer to the same entity/fact as stated in the text. Do NOT use this rule to substitute one entity for another (e.g., replacing a tool with its operator, or an actor with its malware).\n\nExample A Subject specific:\nText: The Magecart attack on British Airways involved purchasing and utilizing an SSL certificate provided by Comodo.\nPrediction: { \"sub\": \"Magecart\", \"rel\": \"uses\", \"obj\": \"SSL certificates (Comodo)\" }\nReasoning: \"Magecart\" and \"Magecart\\'s attack on British Airways\" can be regarded as the same attack entity in this context; the relation holds.\n\nExample B Subject generalization:\nText: Persistent malicious applications on the Google Play platform disseminated the Android.Reputation.1 malware.\nPrediction: { \"sub\": \"Google Play\", \"rel\": \"delivers\", \"obj\": \"Android.Reputation.1\" }\nReasoning: Using the platform \"Google Play\" to summarize \"malicious apps on it\" is a reasonable abstraction for a delivery relation; holds.\n\nExample C Object generalization:\nText: HermeticRansom utilized the Golang GUID library.\nPrediction: { \"sub\": \"HermeticRansom\", \"rel\": \"uses\", \"obj\": \"Golang\" }\nReasoning: \"Golang GUID library\" is a specific library of \"Golang\"; abstracting it to Golang still refers to the same tech stack; holds.\n\nRule 3: Action-Technique Equivalence Rule\n\nA concrete behavioral description of the same technique and a standardized TTP name (e.g., DLL Hollowing, ETW Disable) can be regarded as equivalent in this context.\n\nExample A:\nText: StealthVector disabled the Event Tracing for Windows (ETW) functionality.\nPrediction: { \"sub\": \"StealthVector\", \"rel\": \"employs\", \"obj\": \"ETW Disable\" }\nNote: \"ETW Disable\" is the normalized name of the above behavior; holds.\n\nExample B:\nText: StealthVector injects malicious code into a legitimate DLL.\nPrediction: { \"sub\": \"StealthVector\", \"rel\": \"employs\", \"obj\": \"DLL Hollowing\" }\nNote: DLL hollowing is exactly the above behavior; treat as equivalent.\n\nRule 4: Event-Element Complementarity Rule\n\nWhen two relations describe the same event initiated by the same subject, where one provides \"action + object\" and the other provides \"action + destination\", and the text connects these two parts together, then they are equivalent.\n\nExample:\nText: The last malicious file in the bundle is upload.exe, which uploads the video previously downloaded using download.exe to YouTube.\nRelation A: { \"sub\": \"upload.exe\", \"rel\": \"uploads\", \"obj\": \"videos\" }\nRelation B: { \"sub\": \"upload.exe\", \"rel\": \"exfiltrate-to\", \"obj\": \"YouTube channels\" }\nNote: A focuses on what is uploaded; B focuses on where it is uploaded; together it is \"upload videos to YouTube\", describing the same event.\n\nRule 5: Relation Semantic Equivalence Rule\n\nWhen sub and obj are the same or equivalent, and two rel express the same intent in this context or different facets of the same event, then treat them as equivalent (e.g., has/uses/indicates/delivers under specific contexts).\n\nExample A:\nText: Android.Reputation.1 incorporated the Google Play icon for the purpose of self-disguise.\nPrediction: { \"sub\": \"Android.Reputation.1\", \"rel\": \"uses\", \"obj\": \"Google Play icon\" }\nNote: \"carried for disguise\" and \"uses the icon\" are equivalent in this context.\n\nExample B:\nText: Infection with AZORult occurred after a user downloaded ProtonVPN_win_v1.10.0.exe.\nGround truth: { \"sub\": \"ProtonVPN_win_v1.10.0.exe\", \"rel\": \"indicates\", \"obj\": \"AZORult\" }\nPrediction: { \"sub\": \"ProtonVPN_win_v1.10.0.exe\", \"rel\": \"delivers\", \"obj\": \"AZORult\" }\nNote: From the attacker perspective, the file \"delivers\" AZORult; from the detection perspective, its presence \"indicates\" AZORult; the core fact is consistent.\n\nRule 6: Exclusion and Rejection Rule (Malformed Extractions)\n\nIf sub or obj is not a clear named entity, but a whole sentence/long clause/pronoun (I/you/which, etc.), then the relation is an invalid extraction; during matching it is treated as an incorrect result and cannot count as TP.\n\nTypical errors:\nsub is a long sentence:\n{ \"sub\": \"CVE-2022-22965 and CVE-2022-22963 : technical details CVE-2022-22965 (Spring4Shell, SpringShell)\", \"rel\": \"be\", \"obj\": \"a vulnerability in the Spring Framework that uses ...\" }\n\nobj is a whole sentence:\n{ \"sub\": \"A vulnerable configuration\", \"rel\": \"consist\", \"obj\": \"of: JDK version 9 + Apache Tomcat ... long clause ...\" }\n\nsub is a pronoun:\n{ \"sub\": \"which\", \"rel\": \"make\", \"obj\": \"CVE-2022-22965 a critical threat\" }\n{ \"sub\": \"you\", \"rel\": \"fix\", \"obj\": \"CVE-2022-22963\" }\n\nSuch relations cannot match any correct relation.\n\nRule 7: Placeholder Entity Resolution Rule (Attacker/Attacking placeholders)\n\nWhen encountering placeholder entities such as Attacker(using: X), Attacking(using: Y), Attacking(from: Z), interpret them as \"the attacker/attack activity related to X/Y/Z\", rather than the literal string.\n\nDuring evaluation:\n\n1. Treat the subject as \"the attacker who uses X\" or \"the attack activity that uses Y\".\n2. In the text, look for fused evidence that simultaneously satisfies \"related to X/Y/Z\" and \"performed rel on object obj\".\n3. As long as the text explicitly supports it, judge as TP.\n\nExample A:\nText: The campaign, orchestrated by an unknown actor, leveraged CVE-2021-44228 to gain initial access.\nPrediction: { \"sub\": \"Attacker(using: CVE-2021-44228)\", \"rel\": \"gains-access\", \"obj\": \"target_system\" }\nNote: The text clearly states that \"the attacker using CVE-2021-44228\" gained initial access; holds.\n\nExample B:\nText: A recent wave of attacks utilized the EternalBlue exploit to propagate laterally within networks.\nPrediction: { \"sub\": \"Attacking(using: EternalBlue)\", \"rel\": \"propagates-laterally\", \"obj\": \"networks\" }\nNote: The text directly states that \"the attack activity using EternalBlue\" performed lateral movement; holds.\n\nRule 8: Canonical Relation Validation Rule\n\nIf rel is exactly a canonical Name in GRID_rel_type_definition (e.g., communicates-with, downloads, etc.), it does not have to appear in the original text with exactly the same wording. Use the definition as the standard and judge whether the text semantics match that definition.\n\nExample:\nIn GRID_rel_type_definition, communicates-with is defined as \"describes network communication between two entities\".\nText: analysis revealed network traffic between the infected host and the domain evil.com.\nPrediction: { \"sub\": \"infected host\", \"rel\": \"communicates-with\", \"obj\": \"evil.com\" }\nNote: \"network traffic between\" is network communication; matches the definition; holds.\n\nRule 9: Relationship Hierarchy Inclusion Rule\n\nWhen rel has a parent-child hierarchy (e.g., communicates-with has subclasses downloads/exfiltrate-to/beacons-to), be tolerant of granularity differences.\n\nCase 1 The prediction is more specific (subclass):\nFor example, prediction (A, downloads, B), ground truth (A, communicates-with, B).\nIf the text clearly supports the specific behavior \"download files\", then downloads is TP; if the text only states communication without a download implication, then downloads is over-inference, treated as FP in Precision evaluation and FN in Recall evaluation.\n\nCase 2 The prediction is more general (parent class):\nFor example, prediction (A, communicates-with, B), ground truth (A, downloads, B).\nAs long as the text truly supports the fact \"download\", then \"there is communication\" must be true. The prediction is coarser but correct; it should be judged as TP.\n\nRule 10: Entity Attribute Validation Rule\n\nUsed for alias-of / is-variant-of / is-part-of and other relations in Ground Truth, but this information may be stored on the prediction side in entity attributes (alias/mother_entity) rather than edges.\n\nWhen the ground truth is:\nsub, rel in {alias-of, is-variant-of, is-part-of and other alias/hierarchy types}, obj\nand no equivalent predicted relation can be found:\n\n1. In the predicted entity list, find an entity whose name is equivalent to sub or obj.\n2. Check whether the other entity name appears in its alias or mother_entity.\n3. If it appears, it means the relation has already been represented via an \"attribute\"; judge as TP, and set index_predict to the matched entity index.\n\nExample:\nGround truth: { \"index_truth\": \"truth_relationship2\", \"sub\": \"REvil\", \"rel\": \"alias-of\", \"obj\": \"Sodinokibi\" }\nPredicted entity: { \"index\": \"predict_entity5\", \"name\": \"Sodinokibi\", \"alias\": [\"REvil\", \"Ransom.Sodinokibi\"] }\nThen:\n{ \"index_truth\": \"truth_relationship2\", \"index_predict\": \"predict_entity5\", \"result\": \"TP\" }\n\nRule 11: Entity Alias Equivalence Rule\n\nIf entity A is an alias of entity B (appears in the alias list), then when evaluating any relation, A and B are fully interchangeable, i.e., (A, rel, C) and (B, rel, C) are treated as the same fact.\n\nExample:\nPrediction: { \"index_predict\": \"predict_relationship_10\", \"sub\": \"APT29\", \"rel\": \"uses\", \"obj\": \"Cobalt Strike\" }\nTruth: { \"index_truth\": \"truth_relationship_4\", \"sub\": \"Cozy Bear\", \"rel\": \"uses\", \"obj\": \"Cobalt Strike\" }\nEntity info: { \"name\": \"Cozy Bear\", \"alias\": [\"APT29\", \"The Dukes\"] }\nThen APT29 and Cozy Bear are equivalent; the prediction is TP and matches truth_relationship_4.\n\nRule 12: Entity Hierarchy Inheritance/Induction Rule\n\nIf A\\'s mother_entity contains B, meaning A is a variant/component/instance of B, then:\n\nDownward inheritance: B\\'s capabilities/behaviors are usually also possessed by A.\nUpward induction: A\\'s specific capabilities/behaviors can be generalized as the family capabilities of B.\n\n\u26a0\ufe0f Constraint: This rule applies ONLY when comparing entities that differ only in specificity level (e.g., Cerber v5.0.1 vs Cerber family). It does NOT allow attributing a component\\'s behavior to its parent system unless the text explicitly states so.\n\nWhen comparing (P_sub, rel, obj) and (T_sub, rel, obj), if rel and obj are equivalent but the subjects differ, equivalence can be determined via mother_entity.\n\nExample:\nPrediction: { \"index_predict\": \"predict_relationship_15\", \"sub\": \"Cerber v5.0.1\", \"rel\": \"encrypts\", \"obj\": \".doc files\" }\nTruth: { \"index_truth\": \"truth_relationship_7\", \"sub\": \"Cerber\", \"rel\": \"encrypts\", \"obj\": \".doc files\" }\nEntity: { \"name\": \"Cerber v5.0.1\", \"mother_entity\": [\"Cerber\"] }\nThen Cerber v5.0.1 and Cerber are equivalent for this behavior; the prediction is TP and matches truth_relationship_7.\n\nRule 13: Entity-Attribute Equivalence Rule\n\nWhen the ground-truth object is a \"general description\", while the predicted object is an important attribute of the same thing (brand/vendor name, etc.), and the text explicitly links the two, then treat them as equivalent.\n\nExample:\nText: at one point we observed a legitimate remote admin client tool by NetSupport Ltd being used to install components during these attacks.\nGround truth: Attacker(using: Sodinokibi) uses legitimate remote admin client tool\nPrediction: Sodinokibi ransomware campaign uses NetSupport Ltd\nNote: \"legitimate remote admin client tool\" and \"the remote tool provided by NetSupport Ltd\" refer to the same tool in this context; holds.\n\nRule 14: Multi-Source Information Fusion & Deduction Rule\n\nWhen a single predicted relation or a simple relation chain cannot directly correspond to the ground truth, but combining multiple predicted relations + the text context can prove the fact stated in the ground truth, it can still be regarded as TP.\n\n\u26a0\ufe0f Constraint: The fusion must NOT change the subject of the relation. All fused relations must maintain the same subject, or the subject substitution must be explicitly supported by an alias/coreference in the text.\n\nExample:\nGround truth: (Spring4Shell, exists in, getCachedIntrospectionResults method)\nPrediction: (Spring Framework, consists-of, getCachedIntrospectionResults) and (Spring Framework, has, CVE-2022-22965)\nText: \"By analogy with the infamous Log4Shell threat, the vulnerability was named Spring4Shell.\" and \"The bug exists in the getCachedIntrospectionResults method...\"\nReasoning: Spring4Shell is an alias of CVE-2022-22965; the vulnerability exists in that method; the ground-truth relation can be deduced to hold.\n\nDemonstration examples (specific demonstrations of some rules)\n\n1 General-Specific equivalence example:\nCorrect relation:\nsub: \"Magecart\"\nrel: \"purchased SSL certificates from\"\nobj: \"Comodo\"\n\nRelation for evaluation:\nsub: \"Magecart\\'s attack on British Airways\"\nrel: \"uses\"\nobj: \"SSL certificates (Comodo)\"\n\nNote: The subject abstraction levels differ but both refer to the Magecart attack; \"purchased ... from / uses\" in this context expresses \"leveraging certificates from Comodo\", treated as an equivalent relation.\n\n2 rel semantic equivalence example:\nOriginal relation:\nsub: \"Android.Reputation.1\"\nrel: \"has\"\nobj: \"Google Play icon\"\n\nEvaluation relation:\nsub: \"Android.Reputation.1\"\nrel: \"uses\"\nobj: \"Google Play icon\"\n\nNote: has and uses both describe \"the malware carries and uses the icon for disguise\", and can be regarded as equivalent.\n\n3 rel semantic equivalence example (indicates vs delivers):\nOriginal relation:\n{ \"sub\": \"ProtonVPN_win_v1.10.0.exe\", \"rel\": \"indicates\", \"obj\": \"AZORult\" }\n\nEvaluation relation:\nsub: \"ProtonVPN_win_v1.10.0.exe\"\nrel: \"delivers\"\nobj: \"AZORult\"\n\nNote: Both indicate a strong association between the file and AZORult infection; from the detection perspective it is indicates, from the attacker perspective it is delivers; essentially the same.\n\n4 Exclude self-promotion/introductory relations:\nFor example:\n{ \"sub\": \"Avast Threat Research\", \"rel\": \"published tweet about\", \"obj\": \"HermeticRansom\" }\n{ \"sub\": \"John Doe\", \"rel\": \"published report about\", \"obj\": \"Tomiris malware\" }\n\nNote: Such relations only express \"who published a report/tweet/analyzed some malware\", belonging to publicity/publication behavior; they are not counted in threat intelligence relation evaluation and can be ignored directly.\n\n5 General-Specific example Golang:\nCorrect relation:\n{ \"sub\": \"HermeticRansom\", \"rel\": \"uses\", \"obj\": \"Golang GUID library\" }\n\nEvaluation relation:\n{ \"sub\": \"HermeticRansom\", \"rel\": \"uses\", \"obj\": \"Golang\" }\n\nNote: The GUID library is part of the Golang ecosystem; abstraction differs but the core fact is the same.\n\n6 Invalid extraction examples:\nSuch as:\n{ \"sub\": \"which\", \"rel\": \"make\", \"obj\": \"CVE-2022-22965 a critical threat\" }\n{ \"sub\": \"you\", \"rel\": \"fix\", \"obj\": \"CVE-2022-22963\" }\n{ \"sub\": \"A vulnerable configuration\", \"rel\": \"consist\", \"obj\": \"of: JDK version 9 + Apache Tomcat ... long clause ...\" }\n\nNote: sub/obj are not clear entities or are overly long clauses; they must uniformly be treated as failed extractions and should not match any ground truth.\n\nGRID_rel_type_definition (used to explain what a rel_type specifically expresses, and the meaning intended when some rels use the same wording for rel_type):\n[REL_LIST grouped by theme]\n\nAttack / Compromise:\nexploits, bypasses, malicious-investigates-track-detects, impersonates, targets, compromises, leads-to\n\nData / Payload Movement:\ndrops, downloads, executes, delivers, beacons-to, exfiltrate-to, leaks, communicates-with\n\nInfrastructure / Provisioning:\nresolves-to, hosts, provides\n\nAttribution / Association:\nauthored-by, owns, controls, attributed-to, affiliated-with, cooperates-with\n\nComposition / Capability / State:\nis-part-of, consists-of, has, depends-on, creates-or-generates, modifies-or-removes-or-replaces, uses\n\nClassification / Lineage:\nvariant-of, derived-from, alias-of, compares-to, categorized-as\n\nGeographic:\nlocated-at, originates-from\n\nAnalysis / Defense:\nindicates, mitigates, based-on, research-describes-analysis-of-characterizes-detects\n\nMeta:\nnegation, other\n\n[EXPLAIN1 brief definitions for individual relations]\n\nexploits: uses a specific vulnerability to achieve a malicious objective; a special case of uses\nbypasses: the attacker successfully bypasses defensive measures\nmalicious-investigates-track-detects: malicious one-off reconnaissance, continuous tracking, or detection (e.g., checking for sandbox/debugger)\nimpersonates: impersonates another independent entity to deceive\ntargets: indicates the object that the attack intent is directed at\ncompromises: has successfully broken confidentiality/integrity/availability\nleads-to: the occurrence of A directly causes B (a causal link in the attack chain)\n\ndrops: a local process writes its embedded resources into a local file (local \u2192 local, no network)\ndownloads: retrieves data from a remote source and saves it locally (external \u2192 local)\nexecutes: runs or triggers another entity\ndelivers: at the tactical level, \"brings the malicious payload into the target environment\"\nbeacons-to: periodically sends heartbeat/beacon traffic to C2\nexfiltrate-to: steals/transfers data from the victim side outward to an attacker-specified location\nleaks: publicly or semi-publicly discloses sensitive information or code\ncommunicates-with: only indicates that network communication exists; purpose is unknown or described generically\n\nresolves-to: a domain name is resolved to an IP\nhosts: infrastructure hosts malicious files/sites/C2 services\nprovides: a more abstract \"provides resources/services/capabilities\"; used when it cannot be precisely described with delivers/hosts\n\nauthored-by: who created/developed the entity\nowns: real-world individuals/organizations\\' ownership of infrastructure/tools\ncontrols: software controls the behavior of another process/component\nattributed-to: attributes responsibility for an attack activity to a threat actor\naffiliated-with: social affiliation or cooperation relationships such as organization/employment/membership\ncooperates-with: active cooperation between two peer entities\n\nis-part-of: A is a component part of B\nconsists-of: what subcomponents B is composed of\nhas: what capabilities/characteristics A has (abstract attributes)\ndepends-on: A depends on B to exist or operate normally\ncreates-or-generates: dynamically creates new files/processes/data, etc. at runtime\nmodifies-or-removes-or-replaces: modifies/deletes/replaces other entities or their components\nuses: the subject actively uses the object to achieve a goal (general behavioral relation)\n\nvariant-of: direct evolution or code-variant relationship\nderived-from: inspired by B in ideas/techniques/design, with no direct code inheritance\nalias-of: different names for the same entity; aliases of each other\ncompares-to: purely compares attributes/behaviors of two entities; does not indicate lineage\ncategorized-as: maps an instance to a category/type\n\nlocated-at: current or known geographic location\noriginates-from: place of origin/source (e.g., country/organizational background)\n\nindicates: the presence of an indicator suggests a given threat/entity is likely present\nmitigates: a defensive measure effectively reduces or eliminates a threat\nbased-on: A\\'s creation or conclusion is based on B\\'s information/analysis\nresearch-describes-analysis-of-characterizes-detects: a report/research describes/analyzes/characterizes/detects a threat or target\n\nnegation: explicitly negates the existence of a relationship/attribute/impact\nother: a catch-all label when a relationship exists but cannot be placed into any of the above categories\n\n[EXPLAIN2 distinctions between easily confused relation pairs]\n\nexploits vs uses:\nuse exploits only for \"exploiting a vulnerability\"; otherwise, use uses for general usage behaviors\n\ntargets vs compromises:\ntargets only means \"intends to target\"; compromises means \"has successfully intruded/compromised\"\n\ntargets vs exploits:\ntargets can refer to an industry/organization/system as a whole; exploits only refers to exploiting a specific vulnerability\n\ndrops vs downloads:\ndrops: locally materializes embedded content into a file\ndownloads: pulls content from a remote source to local\n\ndownloads vs exfiltrate-to:\ndownloads: external \u2192 local\nexfiltrate-to: local \u2192 external\n\ndownloads/exfiltrate-to/beacons-to vs communicates-with:\nall three are special cases of communicates-with; if the purpose is known, use the specific one; if the purpose is unknown, then use communicates-with\n\nbeacons-to vs exfiltrate-to:\nbeacons-to emphasizes \"liveness/heartbeat\"; exfiltrate-to emphasizes \"stealing data out\"\n\nleaks vs exfiltrate-to:\nleaks are mostly public/semi-public disclosures; exfiltrate-to is covert, targeted transfer to an attacker-controlled point\n\nhosts vs delivers vs provides:\nhosts: infrastructure-level \"hosting\"; delivers: tactical-level \"delivering into the victim environment\"; provides: coarse-grained \"providing resources/services\"\n\nowns vs controls:\nowns: the subject is a real-world identity; controls: the subject is software controlling other software/processes\n\nauthored-by vs attributed-to vs affiliated-with:\nauthored-by: who wrote/built this thing\nattributed-to: who is believed responsible for the attack activity\naffiliated-with: social relationships such as employment/membership/affiliation; does not directly indicate attack/ownership\n\nis-part-of vs consists-of vs has:\nis-part-of/consists-of: structural composition relationships\nhas: abstract capabilities/attributes, not \"parts\"\n\ndepends-on vs uses:\ndepends-on: static dependency/prerequisite condition\nuses: active usage behavior\n\nvariant-of vs derived-from vs compares-to vs alias-of:\nvariant-of: direct family/variant\nderived-from: source of inspiration/technique\ncompares-to: only comparison\nalias-of: different names but the entity is exactly the same\n\nlocated-at vs originates-from:\nlocated-at: current deployment location\noriginates-from: origin/source (e.g., country/institution)\n\nindicates vs research-describes-analysis-of-characterizes-detects:\nindicates: \"seeing A makes it likely that B exists\"\nthe latter: \"who wrote/analyzed/characterized/detected what\"\n\nmitigates vs bypasses:\nmitigates: defense successfully blocks/reduces\nbypasses: attack successfully bypasses defense\n\nnegation vs other:\nnegation: explicitly states \"does not exist/is not affected/no association\"\nother: a relationship exists but cannot be categorized, or the original text is \"Unknown/Not Applicable\", etc.\n\nWhen you have completely finished, output \\'<Fin>\\'.'\n\n\n"

def get_judge_prompt_bundle(prompt_mode: str = 'grid_judge_fav') -> dict:
    requested = str(prompt_mode or 'grid_judge_fav').strip() or 'grid_judge_fav'
    if requested == 'grid_judge_fav':
        return {
            'requested_mode': requested,
            'canonical_mode': 'grid_judge_fav',
            'precision_prompt_name': 'grid_judge_fav_precision',
            'recall_prompt_name': 'grid_judge_fav_recall',
            'precision_prompt': grid_judge_fav_precision,
            'recall_prompt': grid_judge_fav_recall,
        }
    return {
        'requested_mode': requested,
        'canonical_mode': requested,
        'precision_prompt_name': 'default_precision_prompt',
        'recall_prompt_name': 'default_recall_prompt',
        'precision_prompt': None,
        'recall_prompt': None,
    }


__all__ = [
    'grid_kg_single_prompt_maker_tracerawtext_20260303',
    'grid_kg_single_prompt_maker_tracerawtext',
    'grid_kg_single_prompt_maker_very_simple_20260303',
    'grid_kg_single_prompt_maker_very_simple',
    'grid_kg_sft_reasoning_reconstruction_prompt_20260303',
    'grid_kg_sft_reasoning_reconstruction_prompt',
    'grid_kg_reverse_prompt_maker_20260303',
    'grid_kg_reverse_prompt_maker',
    'prompt_maker_precision_only',
    'grid_judge_fav_precision',
    'grid_judge_fav_recall',
    'get_judge_prompt_bundle',
]
