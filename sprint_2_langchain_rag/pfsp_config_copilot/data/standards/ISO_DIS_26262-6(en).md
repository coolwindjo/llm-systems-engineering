# ISO_DIS_26262-6(en)

- Source PDF: `ISO_DIS_26262-6(en).PDF`
- Total pages: `70`

## Page 1

© ISO 2016
Road vehicles — Functional safety —
Part 6:
Product development at the software level
Véhicules routiers — Sécurité fonctionnelle —
Partie 6: Développement du produit au niveau du logiciel
ICS: 43.040.10
Reference number
ISO/DIS 26262-6:2016(E)
DRAFT INTERNATIONAL STANDARD
ISO/DIS 26262-6
ISO/TC 22/SC 32 Secretariat: JISC
Voting begins on: Voting terminates on:
2016-09-21 2016-12-13
THIS DOCUMENT IS A DRAFT CIRCULATED
FOR COMMENT AND APPROVAL. IT IS
THEREFORE SUBJECT TO CHANGE AND MAY
NOT BE REFERRED TO AS AN INTERNATIONAL
STANDARD UNTIL PUBLISHED AS SUCH.
IN ADDITION TO THEIR EVALUATION AS
BEING ACCEPTABLE FOR INDUSTRIAL,
TECHNOLOGICAL, COMMERCIAL AND
USER PURPOSES, DRAFT INTERNATIONAL
STANDARDS MAY ON OCCASION HAVE TO
BE CONSIDERED IN THE LIGHT OF THEIR
POTENTIAL TO BECOME STANDARDS TO
WHICH REFERENCE MAY BE MADE IN
NATIONAL REGULATIONS.
RECIPIENTS OF THIS DRAFT ARE INVITED
TO SUBMIT , WITH THEIR COMMENTS,
NOTIFICATION OF ANY RELEVANT PATENT
RIGHTS OF WHICH THEY ARE AWARE AND TO
PROVIDE SUPPORTING DOCUMENTATION.
This document is circulated as received from the committee secretariat.
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 2

ISO/DIS 26262-6:2016(E)

ii © ISO 2016 – All rights reserved
COPYRIGHT PROTECTED DOCUMENT
© ISO 2016, Published in Switzerland
All rights reserved. Unless otherwise specified, no part of this publication may be reproduced or utilized otherwise in any form
or by any means, electronic or mechanical, including photocopying, or posting on the internet or an intranet, without prior
written permission. Permission can be requested from either ISO at the address below or ISO’s member body in the country of
the requester.
ISO copyright office
Ch. de Blandonnet 8 • CP 401
CH-1214 Vernier, Geneva, Switzerland
Tel. +41 22 749 01 11
Fax +41 22 749 09 47
copyright@iso.org
www.iso.org
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 3

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 3
Contents 21
Foreword................................................................................................................................................................................... 6 22
Introduction ............................................................................................................................................................................ 7 23
1 Scope ....................................................................................................................................................................................... 9 24
2 Normative references....................................................................................................................................................10 25
3 Terms and definitions ...................................................................................................................................................10 26
4 Requirements for compliance ...................................................................................................................................10 27
5 General topics for the product development at the software level ..........................................................12 28
5.1 Objectives ................................ ................................ ................................ ................................ ................................ .....12 29
5.2 General ................................ ................................ ................................ ................................ ................................ ..........12 30
5.3 Inputs to this clause ................................ ................................ ................................ ................................ .................. 13 31
5.3.1 Prerequisites ................................ ................................ ................................ ................................ ............................ 13 32
5.3.2 Further supporting information ................................ ................................ ................................ ........................ 13 33
5.4 Requirements and recommendations ................................ ................................ ................................ ................. 14 34
5.5 Work products ................................ ................................ ................................ ................................ ............................ 15 35
6 Specification of software safety requirements ..................................................................................................16 36
6.1 Objectives................................ ................................ ................................ ................................ ............................... 16 37
6.2 General................................ ................................ ................................ ................................ ................................ ....16 38
6.3 Inputs to this clause................................ ................................ ................................ ................................ ............ 16 39
6.3.1 Prerequisites ................................ ................................ ................................ ................................ .................... 16 40
6.3.2 Further supporting information ................................ ................................ ................................ ................ 16 41
6.4 Requirements and recommendations ................................ ................................ ................................ ..........16 42
6.5 Work products................................ ................................ ................................ ................................ ...................... 18 43
7 Software architectural design ...................................................................................................................................18 44
7.1 Objectives ................................ ................................ ................................ ................................ ................................ .....18 45
7.2 General ................................ ................................ ................................ ................................ ................................ ..........19 46
7.3 Inputs to this clause ................................ ................................ ................................ ................................ .................. 19 47
7.3.1 Prerequisites ................................ ................................ ................................ ................................ ............................ 19 48
7.3.2 Further supporting information ................................ ................................ ................................ ........................ 19 49
7.4 Requirements and recommendations ................................ ................................ ................................ ................. 19 50
7.5 Work products ................................ ................................ ................................ ................................ ............................ 26 51
8 Software unit design and implementation.....................................................................................................26 52
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 4

ISO/DIS 26262-6:2016(E)
4 © ISO 2016 – All rights reserved
8.1 Objectives ................................ ................................ ................................ ................................ ................................ .....26 53
8.2 General ................................ ................................ ................................ ................................ ................................ ..........26 54
8.3 Inputs to this clause ................................ ................................ ................................ ................................ .................. 27 55
8.3.1 Prerequisites ................................ ................................ ................................ ................................ ............................ 27 56
8.3.2 Further supporting information ................................ ................................ ................................ ........................ 27 57
8.4 Requirements and recommendations ................................ ................................ ................................ ................. 27 58
8.5 Work products ................................ ................................ ................................ ................................ ............................ 29 59
9 Software unit verification ......................................................................................................................................29 60
9.1 Objectives ................................ ................................ ................................ ................................ ................................ .....29 61
9.2 General ................................ ................................ ................................ ................................ ................................ ..........29 62
9.3 Inputs to this clause ................................ ................................ ................................ ................................ .................. 29 63
9.3.1 Prerequisites ................................ ................................ ................................ ................................ ............................ 29 64
9.3.2 Further supporting information ................................ ................................ ................................ ........................ 30 65
9.4 Requirements and recommendations ................................ ................................ ................................ ................. 30 66
9.5 Work products ................................ ................................ ................................ ................................ ............................ 33 67
10 Software integration and verification..........................................................................................................33 68
10.1 Objectives................................ ................................ ................................ ................................ ................................ ...33 69
10.2 General ................................ ................................ ................................ ................................ ................................ ........33 70
10.3 Inputs to this clause ................................ ................................ ................................ ................................ ................ 34 71
10.3.1 Prerequisites ................................ ................................ ................................ ................................ ......................... 34 72
10.3.2 Further supporting information................................ ................................ ................................ ...................... 34 73
10.4 Requirements and recommendations................................ ................................ ................................ ............... 34 74
10.5 Work products................................ ................................ ................................ ................................ .......................... 37 75
11 Testing of the embedded software ................................................................................................................37 76
11.1 Objective................................ ................................ ................................ ................................ ................................ .....37 77
11.2 General ................................ ................................ ................................ ................................ ................................ ........37 78
11.3 Inputs to this clause ................................ ................................ ................................ ................................ ................ 38 79
11.3.1 Prerequisites ................................ ................................ ................................ ................................ ......................... 38 80
11.3.2 Further supporting information................................ ................................ ................................ ...................... 38 81
11.4 Requirements and recommendations................................ ................................ ................................ ............... 38 82
11.5 Work products................................ ................................ ................................ ................................ .......................... 39 83
Annex A (informative) Overview of and workflow of management of product development at the 84
software level........................................................................................................................................................................40 85
Annex B (informative) Model-based development approaches ....................................................................42 86
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 5

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 5
Annex C (normative) Software configuration ........................................................................................................46 87
Annex D (informative) Freedom from interference between software elements.................................53 88
Annex E (informative) Application of safety analyses and analyses of dependent failures at the 89
software architectural level ...........................................................................................................................................56 90
E.1 Objectives ................................ ................................ ................................ ................................ ................................ .....56 91
E.2 General ................................ ................................ ................................ ................................ ................................ ..........56 92
E.3 Topics for performing analyses................................ ................................ ................................ ............................. 60 93
E.4 Mitigation strategy for weaknesses identified during safety and dependent failure analyses.........67 94
Bibliography ..........................................................................................................................................................................70 95
96
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 6

ISO/DIS 26262-6:2016(E)
6 © ISO 2016 – All rights reserved

Foreword 97
ISO (the International Organization for Standardization) is a worldwide federation of nati onal 98
standards bodies (ISO member bodies). The work of preparing International Standards is normally 99
carried out through ISO technical committees. Each member body interested in a subject for which a 100
technical committee has been established has the right t o be represented on that committee. 101
International organizations, governmental and non-governmental, in liaison with ISO, also take part in 102
the work. ISO collaborates closely with the International Electrotechnical Commission (IEC) on all 103
matters of electrotechnical standardization. 104
The procedures used to develop this document and those intended for its further maintenance are 105
described in the ISO/IEC Directives, Part 1. In particular the different approval criteria needed for the 106
different types of ISO documents should be noted. This document was drafted in accordance with the 107
editorial rules of the ISO/IEC Directives, Part 2. www.iso.org/directives 108
Attention is drawn to the possibility that some of the elements of this document may be th e subject of 109
patent rights. ISO shall not be held responsible for identifying any or all such patent rights. Details of 110
any patent rights identified during the development of the document will be in the Introduction and/or 111
on the ISO list of patent declara tions received. www.iso.org/patents 112
Any trade name used in this document is information given for the convenience of users and does not 113
constitute an endorsement. 114
For an explanation on the meaning of ISO specific terms and expressions related to conformity 115
assessment, as well as information about ISO's adherence to the WTO principles in the Technical 116
Barriers to Trade (TBT) see the following URL: Foreword - Supplementary information 117
The committee responsible for this document is ISO/TC22/SC32 Electrical and electronic components 118
and general sys tem aspects. 119
This second edition cancels and replaces the first edition which has been technically revised. 120
A list of all parts in the ISO 26262 series can be found on the ISO website. 121
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 7

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 7

Introduction 122
ISO 26262 is the adaptation of IEC 61508 to address the se ctor specific needs of electrical and/or 123
electronic (E/E) systems within road vehicles. 124
This adaptation applies to all activities during the safety lifecycle of safety-related systems comprised of 125
electrical, electronic and software components. 126
Safety is o ne of the key issues in the development of road vehicles. Development and integration of 127
automotive functionalities strengthen the need for functional safety and the need to provide evidence 128
that functional safety objectives are satisfied. 129
With the trend o f increasing technological complexity, software content and mechatronic 130
implementation, among others there are increasing risks from systematic failures and random 131
hardware failures, these being considered within the scope of functional safety. ISO 26262 includes 132
guidance to mitigate these risks by providing appropriate requirements and processes. 133
To achieve functional safety, ISO 26262: 134
a) provides a reference for the automotive safety lifecycle and supports the tailoring of the 135
activities to be performed during the lifecycle phases, i.e., development, production, operation, 136
service, and decommissioning; 137
b) provides an automotive-specific risk-based approach to determine integrity levels [Automotive 138
Safety Integrity Levels (ASIL)]; 139
c) uses ASILs to specify which of the requirements of ISO 26262 are applicable to avoid 140
unreasonable residual risk; 141
d) provides requirements for functional safety management, verification, validation and 142
confirmation measures; and 143
e) provides requirements for relations with suppliers. 144
ISO 26262 is concerned with functional safety of E/E systems that is achieved through safety measures 145
including safety mechanisms. It also provides a framework within which safety-related systems based 146
on other technologies (e.g. mechanical, hydraulic and pneumatic ) can be considered. 147
The achievement of functional safety is influenced by the development process (including such 148
activities as requirements specification, design, implementation, integration, verification, validation and 149
configuration), the production and service processes and the management processes. 150
Safety is intertwined with common function-oriented and quality-oriented activities and work products. 151
ISO 26262 addresses the safety -related aspects of these activities and work products. 152
Figure 1 shows t he overall structure of ISO 26262. ISO 26262 is based upon a V -model as a reference 153
process model for the different phases of product development. Within the figure: 154
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 8

ISO/DIS 26262-6:2016(E)
8 © ISO 2016 – All rights reserved

 the shaded “V”s represent the interconnection among ISO 26262-3, ISO 26262-4, ISO 26262-5, 155
ISO 26262-6 and ISO 26262-7; for motorcycles ISO 26262-12 clauses 6 supplements ISO 26262 -3 156
and clause 7, 8 and 9 supplements Part 4. 157
 the specific clauses are indicated in the following manner: “m-n”, where “m” represents the number 158
of the particular part and “n” indicates the number of the clause within that part. 159
EXAMPLE “2-6” represents Clause 6 of ISO 26262-2. 160
161
Figure 1 — Overview of ISO 26262 162
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 9

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 9

Road Vehicles — Functional Safety — Part 6: Product 163
development at the software level 164
1 Scope 165
ISO 26262 is intended to be applied to safety-related systems that include one or more electrical and/or 166
electronic (E/E) systems and that are installed in series production road vehicles, excluding mopeds . 167
ISO 26262 does not address unique E/E systems in special vehicles such as E/E systems designed for 168
drivers with disabilities. 169
NOTE Other dedicated application-specific safety standards exist and may complement ISO 26262 or vice versa. 170
Systems and their components released for production, or systems and their components already under 171
development prior to the publication date of this edition of ISO 26262, are exempted from the scope of 172
this edition. For further development or alt erations based on systems and their components released 173
for production prior to the publication of this edition of ISO 26262, only the modifications will be 174
developed in accordance with this edition of ISO 26262. This edition of ISO 26262 addresses integration 175
of existing systems not developed according to this edition of ISO 26262 and systems developed 176
according to this edition of ISO 26262 by tailoring the safety lifecycle. 177
ISO 26262 addresses possible hazards caused by malfunctioning behaviour of safety -related E/E 178
systems, including interaction of these systems. It does not address hazards related to electric shock, 179
fire, smoke, heat, radiation, toxicity, flammability, reactivity, corrosion, release of energy and similar 180
hazards, unless directly caused b y malfunctioning behaviour of safety -related E/E systems. 181
ISO 26262 does not address the nominal performance of E/E systems, even if functional performance 182
standards exist for these systems (e.g. active and passive safety systems, brake systems, adaptive cruise 183
control). 184
ISO 26262 describes a framework for functional safety to assist the development of safety-related E/E 185
systems. This framework is intended to be used to integrate functional safety activities into a company-186
specific development framework. So me requirements have a clear technical focus to implement 187
functional safety into a product; others address the development process and can therefore be seen as 188
process requirements in order to demonstrate the capability of an organization with respect to 189
functional safety. 190
This part of ISO 26262 specifies the requirements for product development at the software level for 191
automotive applications, including the following: 192
 general topics for product development at the software level; 193
 specification of the software safety requirements; 194
 software architectural design; 195
 software unit design and implementation; 196
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 10

ISO/DIS 26262-6:2016(E)
10 © ISO 2016 – All rights reserved

 software unit verification; 197
 software integration and verification; and 198
 testing of the embedded software. 199
It also specifies requirements associated with the use of configurable software. 200
2 Normative references 201
The following documents, in whole or in part, are normatively referenced in this document and are 202
indispensable for its application. For dated references, only the edition cited applies. For undated 203
references, the latest edition of the referenced document (including any amendments) applies. 204
ISO 26262-1:2018, Road Vehicles — Functional Safety— Part 1: Vocabulary 205
ISO 26262 -2:2018, Road Vehicles — Functional Safety— Part 2: Management of functional safety 206
ISO 26262-3:2018, Road vehicles — Functional safety — Part 3: Concept phase 207
ISO 26262-4:2018, Road vehicles — Functional safety — Part 4: Product development at the system level 208
ISO 26262-5:2018, Road vehicles — Functional safety — Part 5: Product developm ent at the hardware 209
level 210
ISO 26262-7:2018, Road vehicles — Functional safety — Part 7: Production, operation, service and 211
decommissioning 212
ISO 26262-8:2018, Road vehicles — Functional safety — Part 8: Supporting processes 213
ISO 26262-9:2018, Road vehicles — Functional safety — Part 9: Automotive Safety Integrity Level (ASIL)-214
oriented and safety-oriented analyses 215
3 Terms and definitions 216
For the purposes of this document, the terms, definitions and abbre viated terms given in 217
ISO 26262-1:2018 apply. 218
4 Requirements for compliance 219
4.1 Purpose 220
This clause describes how: 221
a) to achieve compliance with ISO 26262; 222
b) to interpret the tables used in ISO 26262; and 223
c) to interpret the applicability of each clause, depending on the relevant ASIL(s). 224
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 11

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 11

4.2 General requirements 225
When claiming compliance with ISO 26262, each requirement shall be met, unless one of the following 226
applies: 227
a) tailoring of the safety activities in accordance with ISO 26262-2 has been performed that shows that 228
the requirement does not apply, or 229
b) a rationale is available that the non-compliance is acceptable and the rationale has been assessed in 230
accordance with ISO 26262-2, for example as part of the functional safety assessment. 231
Informative content, including notes and examples, is only for guidance in understanding, or for 232
clarification of the associated requirement, and shall not be interpreted as a requirement itself or as 233
complete or exhaustive. 234
The results of safety activities are given as work products. “Prerequisites” are information which shall 235
be available as work products of a previous phase. Given that certain requirements of a clause are 236
ASIL-dependent or may be tailored, certain work products may not be needed as prerequisites. 237
“Further supporting information” is information that can be considered, but which in some cases is not 238
required by ISO 26262 as a work product of a previous phase and which may be made available by 239
external sources that are different from the persons or organizations responsible for the functional 240
safety activities. 241
4.3 Interpretations of tables 242
Tables are normative or informative depending on their context. The different methods listed in a table 243
contribute to the level of confidence in achieving compliance with the corresponding requirement. Each 244
method in a table is either 245
a) a consecutive entry (marked by a sequence number in the leftmost column, e.g. 1, 2, 3), or 246
b) an alternative entry (marked by a number followed by a letter in the leftmost column, e.g. 2a, 2b, 247
2c). 248
For consecutive entries, all listed highly recommended and recommended methods in accordance with 249
the ASIL apply. It is allowed to substitute a highly recommended or recommended method by other 250
one(s) not listed in the table, but a rationale shall be given that these comply with the corresponding 251
requirement. A recommended method may be omitted, but a rationale why this method is omitted shall 252
be given. 253
For alternative entries, an appropriate combination of methods shall be applied in accordance with the 254
ASIL indicated, independent of whether they are listed in the table or not. If methods are listed with 255
different degrees of recommendation for an ASIL, the methods with the higher recommendation should 256
be preferred. A rationale shall be given that the selected combination of methods or even a selected 257
single method complies with t he corresponding requirement. 258
NOTE A rationale based on the methods listed in the table is sufficient. However, this does not imply a bias 259
for or against methods not listed in the table. 260
For each method, the degree of recommendation to use the corresponding method depends on the ASIL 261
and is categorized as follows: 262
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 12

ISO/DIS 26262-6:2016(E)
12 © ISO 2016 – All rights reserved

 “++” indicates that the method is highly recommended for the identified ASIL; 263
 “+” indicates that the method is recommended for the identified ASIL; 264
 “o” indicates that the method has no recommendat ion for or against its usage for the identified 265
ASIL. 266
4.4 ASIL-dependent requirements and recommendations 267
The requirements or recommendations of each subclause shall be met for ASIL A, B, C and D, if not 268
stated otherwise. These requirements and recommendations refer to the ASIL of the safety goal. If ASIL 269
decomposition has been performed at an earlier stage of development, in accordance with 270
ISO 26262-9:2018, Clause 5, the ASIL resulting from the decomposition shall be met. 271
If an ASIL is given in parentheses i n ISO 26262, the corresponding subclause shall be considered as a 272
recommendation rather than a requirement for this ASIL. This has no link with the parenthesis notation 273
related to ASIL decomposition. 274
4.5 Adaptation for motorcycles 275
For items or elements for which requirements of ISO 26262 -12 are applicable, the requirements of 276
ISO 26262-12 supersede the corresponding requirements in this Part of ISO 26262. 277
4.6 Adaptation for Trucks, Buses, Trailers and Semitrailers 278
Content that is intended to be unique for Trucks, Buses, Trailers and Semitrailers (T&B) is indicated as 279
such. 280
5 General topics for the product development at the software level 281
5.1 Objectives 282
The objective of this clause is to give an overview of the product development at the software level and 283
to provide topics specific to software development . 284
5.2 General 285
The reference phase model for the development of software is given in Figure 2. Details concerning the 286
treatment of configurable software are pro vided in Annex C. 287
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 13

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 13

288
NOTE Within the figure, the specific clauses of each part of ISO 26262 are indicated in the following manner: 289
“m-n”, where “m” represents the number of the part and “n” indicates the number of the clause, e.g. “4 -7” 290
represents Clause 7 of ISO 26262-4. 291
Figure 2 — Reference phase model for the software development 292
NOTE Methods from Agile Software Development may also be suitable for the development of safety -related 293
software but Agile processes cannot be used to avoid requirements for documentation, process or product rigour. 294
EXAMPLE Improve quality and testability of requirements by Test Driven Development. 295
In order to be able to develop software, specific topics are addressed in this clause concerning the 296
modelling and/or programming lang uages to be used, and the application of guidelines and tools. 297
NOTE Tools used for software development can include tools other than software tools. 298
EXAMPLE Tools used for testing phases. 299
5.3 Inputs to this clause 300
5.3.1 Prerequisites 301
The following information shall be available: 302
 (none) 303
5.3.2 Further supporting information 304
The following information can be considered: 305
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 14

ISO/DIS 26262-6:2016(E)
14 © ISO 2016 – All rights reserved

 qualified software tools available (see ISO 26262-8:2018, Clause 11); 306
 design and coding guidelines for modelling and programming languages (from an external source); 307
 guidelines for the application of methods (from an external source); and 308
 guidelines for the application of tools (from an external source). 309
5.4 Requirements and recommendations 310
5.4.1 The software development process for the software of an item, including lifecycle phases, 311
methods, languages and tools, shall be consistent across all the subphases of the software lifecycle and 312
be compatible with the system and hardware development phases, such that the required data can be 313
used consistently between phases . 314
NOTE 1 The sequencing of phases, tasks and activities, including iteration steps, for the software of an item 315
intends to ensure the consistency of the correspondi ng work products with the product development at the 316
hardware level (see ISO 26262-5) and the product development at the system level (see ISO 26262-4). 317
EXAMPLE Agile methods such as continuous integration based on an automated build system can support 318
consistency of subphases and facilitate regression tests. Such a build system typically performs code generation, 319
compiling and linking, static and semantic code analysis, documentation generation, testing and packaging. It 320
allows, subject to tool chain and tool configuration, repeatable and, after changes, comparable production of 321
software, documentation and test results. However the required process rigour cannot be ignored. 322
NOTE 2 The software tool criteria evaluation report (see ISO 26262-8, 11.5.1) or the software tool qualification 323
report (see ISO 26262-8, 11.5.2) can provide input to the tool application guidelines. 324
5.4.2 The criteria that shall be considered when selecting a suitable modelling or programming 325
language are: 326
a) an unambiguous and understan dable definition; 327
EXAMPLE Unambiguous definition of s yntax and semantics or restriction to configuration of the 328
development environment. 329
b) suitability for the design or/and implementation of embedded real time software including runtime 330
error handling; 331
c) support the achievement of modularity and abstraction; 332
d) support the use of structured constructs; and 333
e) suitability for specifying and managing safety requirements according to ISO 26262 -8 Clause 6, if 334
modelling is used for requirements engineering and manage ment. 335
NOTE 2 Assembly languages can be used for those parts of the software where the use of high-level programming 336
languages is not appropriate, such as low -level software with interfaces to the hardware, interrupt handlers, or 337
time-critical algorithms. 338
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 15

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 15

5.4.3 Criteria for suitable modelling or programming languages (see 5.4. 2) that are not sufficiently 339
addressed by the language itself shall be covered by the corresponding guidelines, or by the 340
development environment , considering the topics listed in Table 1. 341
EXAMPLE 1 MISRA C [3] is a coding guideline for the programming language C and includes guidance for 342
automatically generated code. 343
EXAMPLE 2 In the case of model based development with automatic code generation, g uidelines can be 344
applied at the model level as well as the code level. Appropriate modelling style guides including MISRA AC series 345
can be considered. Style guides for commercial tools are also possible guidelines . 346
NOTE 3 Existing coding guidelines and modelling guidelines can be modified for a specific item development. 347
Table 1 — Topics to be covered by modelling and coding guidelines 348
Topics ASIL
A B C D
1a Enforcement of low complexity a
++ ++ ++ ++
1b Use of language subsets b
++ ++ ++ ++
1c Enforcement of strong typing c
++ ++ ++ ++
1d Use of defensive implementation techniques d
+ + ++ ++
1e Use of well-trusted design principles e
+ + ++ ++
1f Use of unambiguous graphical representation + ++ ++ ++
1g Use of style guides + ++ ++ ++
1h Use of naming conventions ++ ++ ++ ++
1i Representation of concurrency aspects f + + + +
a An appropriate compromise of this topic with other requirements of this part of ISO 26262 may be required.
b The objectives of topic 1b include:
 Exclusion of ambiguously -defined language constructs which may be interpreted differently by different modellers,
programmers, code generators or compilers.
 Exclusion of language constructs w hich from experience easily lead to mistakes, for example assignments in
conditions or identical naming of local and global variables.
 Exclusion of language constructs which could result in unhandled run -time errors.
c The objective of topic 1c is to impos e principles of strong typing where these are not inherent in the language.
d Examples of defensive implementation techniques:
 Verify the divisor before a division operation (different from zero or in a specific range).
 Check an identifier passed by param eter to verify that the calling function is the intended caller.
 Use the “default” in switch cases to detect an error.
e Verification of the validity of the underlying assumptions, boundaries and conditions of application may be required.
f Concurrency of processes or tasks may be a topic when executing software in a multi -core or multi-processor runtime
environment
5.5 Work products 349
5.5.1 Design and coding guidelines for modelling and programming languages resulting from 350
requirements 5.4. 2 and 5.4.3. 351
5.5.2 Tool application guidelines resulting from requirements 5.4.1 to 5.4.3. 352
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 16

ISO/DIS 26262-6:2016(E)
16 © ISO 2016 – All rights reserved

6 Specification of software safety requirements 353
6.1 Objectives 354
The objectives of this subphase are: 355
a) to specify the software safety requirements. They are derived from the technical safety concept and 356
the system design specification; 357
b) to refine the hardware -software interface requirements initiate d in ISO 26262 -4:2018, Clause 7; 358
and 359
c) to verify that the software safety requirements and the hardware-software interface requirements 360
are consistent with the technical safety concept and the system design specification. 361
6.2 General 362
The technical safety requirements are refined and allocated to hardware and software during the 363
system architectural design phase given in ISO 26262-4:2018, Clause 7. The specification of the 364
software safety requirements considers in particular constraints of the hardware and the impact of 365
these constr aints on the sof tware. This sub phase includes the specification of software safety 366
requirements to support the subsequent design phases. 367
6.3 Inputs to this clause 368
6.3.1 Prerequisites 369
The following information shall be available: 370
 technical safety concept in accordance with ISO 26262-4:2018, 7.5.1; 371
 system architectural design specification in accordance with ISO 26262-4:2018, 7.5.2; and 372
 hardware -software interface (HSI) specification in accordance with ISO 26262-4:2018, 7.5.3. 373
6.3.2 Further supporting information 374
The following information can be considered: 375
 hardware design specification (see ISO 26262-5:2018, 7.5.1); 376
 specification of non -safety-related functions and properties of the software (from an external 377
source); and 378
 guidelines for the application of methods (fro m an external source). 379
6.4 Requirements and recommendations 380
6.4.1 The software safety requirements shall be derived considering the required safety -related 381
functionalities and properties of the software, whose failures could lead to violation of a technical safety 382
requirement allocated to software . 383
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 17

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 17

NOTE 1 The software safety requirements are either derived directly from the technical safety requirements 384
allocated to software, or are requirements for software functions and properties that if not fulfilled could lead to a 385
violation of the technical safety requirements allocated to software. 386
EXAMPLE 1 Safety-related functionality of the software can be: 387
 functions that enable the safe execution of a nominal function, 388
 functions that enable the system to achieve or maintain a safe state or degraded state, 389
 functions related to the detection, indication and mitigation of faults of safety-related hardware elements, 390
 self-test or monitoring functions related to the detection, indication and mitigation failures in the operating 391
system, basic software or the application software itself , 392
 functions related to on-board and off-board tests during production, operation, service and decommissioning, 393
 functions that allow modifications of the software during production and service, or 394
 functions related to performance or time-critical operations. 395
EXAMPLE 2 Safety-related properties include robustness against erroneous inputs, independence or freedom 396
from interference between different functionalities, fault tolerance capabilities of the software. 397
NOTE 2 Safety analyses can be used to identify additional software safety requirements or provide evidence for 398
their achievement. 399
6.4.2 Specification of the software safety requirements derived from the technical safety concept and 400
the system design in accordance with ISO 26262-4:2018, 7.4.1 and 7.4.5 shall consider: 401
a) the specification and management of safety requirements in accordance with ISO 26262-8:2018, 402
Clause 6; 403
b) the specified system and hardware configurations; 404
EXAMPLE 1Configuration parameters can include gain control, band pass frequency and clock prescaler. 405
c) the hardware -software interface specification; 406
d) the relevant requirements of the hardware design specification; 407
e) the timing constraints; 408
EXAMPLE 2Execution or reaction time derived from the required response time at the system level. 409
f) the external interfaces; and 410
EXAMPLE 3Communication and user interfaces. 411
g) each operating mode and each transition between the operating modes of the vehicle, the system, or 412
the hardware, having a n impact on the software. 413
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 18

ISO/DIS 26262-6:2016(E)
18 © ISO 2016 – All rights reserved

EXAMPLE 4 Operating modes include shut-off or sleep , initialization, normal operation, degraded and 414
advanced modes for testing or flash programming. 415
6.4.3 If ASIL decomposition is applied to the software safety requirements, ISO 26262-9:2018, 416
Clause 5, shall be complied with. 417
6.4.4 The hardware-software interface specification initiated in ISO 26262-4:2018, Clause 7, shall be 418
refined sufficiently to allow for the correct control and usage of the hardware by the software, and shall 419
describe each safety -related dependency between hardware and software. 420
6.4.5 If other functions in addition to those functions for which safety requirements are specified in 421
6.4.1 are carried out by the embedded software, a specification of these functions and their properties 422
in accordance with the applied quality management system shall be available. 423
6.4.6 The refined hardware -software interface specification shall be verified jointly by the persons 424
responsible for the system, hardware and software devel opment. 425
6.4.7 The software safety requirements and the refined requirements of the hardware -software 426
interface specification shall be verified in accordance with ISO 26262-8:2018, Clauses 6 and 9, to 427
provide evidence for their: 428
a) compliance and consistency with the technical safety requirements; 429
b) compliance with the system design; and 430
c) consistency with the hardware -software interface. 431
6.5 Work products 432
6.5.1 Software safety requirements specification resulting from requirements 6.4.1 to 6.4.3 and 433
6.4.5. 434
6.5.2 Hardware-software interface (HSI) specification (refined) resulting from requirement 6.4.4. 435
NOTE This work product refers to the same work product as given in ISO 26262-5:2018, 6.5.2 436
6.5.3 Software verification report resulting from requirements 6.4. 6 and 6.4.7. 437
7 Software architectural design 438
7.1 Objectives 439
The objective of this subphase are : 440
a) to develop a software architectural design that satisfies the software safety requirements and the 441
non-safety-related requirement s; 442
b) to verify that the software architectural design is suitable to satisfy the software safety 443
requirements with the required ASIL; and 444
c) to support the implementation of the software. 445
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 19

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 19

7.2 General 446
The software architectural design represents the software architectural elements and their interactions 447
in a hierarchical structure. Static aspects, such as interfaces between the software components, as well 448
as dynamic aspects, such as process sequences and timing behaviour are described. 449
NOTE The software architectural design is not necessarily limited to one microcontroller or ECU. The software 450
architecture for each microcontroller is also addressed by this subclause. 451
A software architectural design must be able to satisfy both software safety requirements as well as the 452
non-safety-related requirements. Hence in this subphase safety -related and non -safety-related 453
requirements are handled within one development process. 454
The software architectural design provides the means to implement both the software requirements 455
and the software safety requirements with the required integrity and to manage the complexity of the 456
detailed design and the implementation of the software. 457
7.3 Inputs to this clause 458
7.3.1 Prerequisites 459
The following information shall be available: 460
 design and coding guidelines for modelling and programming languages in accordance with 5.5. 1; 461
 hardware -software interface (HSI) specification (refined) in accordance with 6.5.2; and 462
 software safety requirements specification in ac cordance with 6.5.1 . 463
7.3.2 Further supporting information 464
The following information can be considered: 465
 technical safety concept (see ISO 26262-4:2018, 7.5.1); 466
 system architectural design specification (see ISO 26262-4:2018, 7.5.2); 467
 qualified software components available (see ISO 26262-8:2018, Clause 12); 468
 tool application guidelin es (see 5.5.2); and 469
 guidelines for the application of methods (from an external source). 470
7.4 Requirements and recommendations 471
7.4.1 To avoid systematic faults in the software archi tectural design and in the subsequent 472
development activities, the description of the software architectural design shall address the following 473
characteristics supported by appropriate guidelines and by notations for software architectural design 474
as listed in Table 2: 475
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 20

ISO/DIS 26262-6:2016(E)
20 © ISO 2016 – All rights reserved

a) unambiguity, consistency, comprehensibility, maintainability and verifiability of the 476
architectural design; 477
b) modularity and abstraction; and 478
NOTE Abstraction can be supported by using hierarchical structures, grouping schemes or views to c over 479
static, dynamic or deployment aspects of an architectural design. 480
c) low complexity. 481
Table 2 — Notations for software architectural design 482
Notations ASIL
A B C D
1a Natural language a ++ ++ ++ ++
1b Informal notations ++ ++ + +
1c Semi-formal notations b + + ++ ++
1d Formal notations + + + +
a Natural language can complement the use of notations for example where some topics are more readily expressed in
natural language, or providing explanation and rationale for decisions captured in the notation
b Semi-formal notation s include pseudocode or modelling with UML ®, SysMLTM, Simulink® or Stateflow ®, structured
sentences or requirement patterns with controlled vocabulary
483
7.4.2 During the development of the software architectural design the following shall be considered: 484
a) the verifiability of the software architectural design; 485
NOTE This implies bi-directional traceability between the software architectural design and the software 486
safety requirements. 487
b) the suitability for configurable software; 488
c) the feasibility for the design and implementation of the software units; 489
d) the testability of the software architecture du ring software integration testing; and 490
e) the maintainability of the software architectural design. 491
7.4.3 In order to avoid systematic faults, the software architectural design itself shall exhibit the 492
following properties by use of the principles listed in T able 3: 493
a) comprehensibility; 494
b) modularity; 495
c) encapsulation; and 496
d) simplicity. 497
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 21

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 21

Table 3 — Principles for software architectural design 498
Principles ASIL
A B C D
1a Appropriate hierarchical structure of the software components ++ ++ ++ ++
1b Restricted size and complexity of software components a
++ ++ ++ ++
1c Restricted size of interfaces a
+ + + ++
1d Strong cohesion within each software component b
+ ++ ++ ++
1e Loose coupling between software components a, b, c
+ ++ ++ ++
1f Appropriate scheduling properties ++ ++ ++ ++
1g Restricted use of interrupts a
+ + + ++
1h Use of only priority-based interrupts + + + ++
1i Appropriate management of shared resources d
++ ++ ++ ++
a In principle 1b, 1c, 1e and 1g “restricted” means to minimize in balance with other design considerations.
b Principle 1d and 1e can, for example, be achieved by separation of concerns which refers to the ability to identify,
encapsulate, and manipulate those parts of software that are relevant to a particular concept, goal, task, or purpose.
c Principle 1e addresses the limitation of dependencies between software components without an intende d functional
relationship.
d
Applies for shared hardw are resources as w ell as shared softw are resources. Such resource management can be
implemented in softw are or hardw are and includes mechanisms that prevents conflicting access to shared resources as wel l
as mechanisms that detects conflicting access to shared resources and handle them.
NOTE 1 An appropriate compromise between the principles listed in Table 3 can be necessary since the 499
principles are not mutually exclusive. 500
NOTE 2 Indicators for high complexity can be: 501
 highly branched control or data flow, 502
 excessive number of requirements allocated to single design elements, 503
 excessive number of interfaces of one design element or interactions between design elements, 504
 complex types or excessive number of parameters, 505
 excessive number of global variables, 506
 difficulties to provide evidence for suitability and completeness of error detection and handling, 507
 difficulties to achieve the required test coverage, or 508
 comprehensibility only for few experts or only by project participants. 509
NOTE 3 These properties and principles also apply to software routines (e.g. service routines for interrupt 510
handling). 511
7.4.4 The software architectural design shall be developed down to the level where the software units 512
are identified. 513
7.4.5 The software architectural design shall describe: 514
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 22

ISO/DIS 26262-6:2016(E)
22 © ISO 2016 – All rights reserved

a) the static design aspects of the software architectural elements ; and 515
NOTE 1 Static design aspects address: 516
 the software structure including its hierarchical levels; 517
 the data types and their characteristics; 518
 the external interfaces of the software components; 519
 the external interfaces of the software; 520
 the data flow at global variables; and 521
 the constraints including the scope of the architecture and external dependencies. 522
NOTE 2 In the case of model-based development, modelling the structure is an inherent part of the overall 523
modelling activities. 524
b) the dynamic design aspects of the software architectural elements . 525
NOTE 1 Dynamic design aspects address: 526
 the functional chain of events and behaviour; 527
 the logical sequence of data processing; 528
 the control flow and concurrency of processes; 529
 the data flow between the software components; 530
 the data flow at external interfaces; and 531
 the temporal constraints. 532
NOTE 2 To determine the dynamic behaviour (e.g. of tasks, time slices and interrupts) the different 533
operating states (e.g. power-up, shut-down, normal operation, calibration and diagnosis) are considered. 534
NOTE 3 To describe the dynamic behaviour (e.g. of tasks, time slices and interrupts) the communication 535
relationships and their allocation to the system hardware (e.g. CPU and communication channels) are 536
specified. 537
7.4.6 The software safety requirements shall be allocated to the software components. As a result, 538
each software component shall be developed in compliance with the highest ASIL of any of the 539
requirements allocated to it. 540
NOTE Following this allocation, further refinement of the software safety requirements can be necessary. 541
7.4.7 If an already existing software architectural element is used without modifications in order to 542
meet the assigned safety requirements without being already developed according to ISO 26262, then it 543
shall be qualified in accordance with ISO 26262-8:2018, Clause 12. 544
NOTE The use of qualified software components does not affect the applicability of Clauses 10 and 11. However, 545
some activities described in Clauses 8 and 9 can be omitted. 546
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 23

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 23

NOTE The suitability for reuse of so ftware elements already developed according to ISO 26262 is performed 547
during the verification of the software architectural design. 548
7.4.8 If the embedded software has to implement software components of different ASILs, or safety -549
related and non-safety-related software components, then all of the embedded software shall be treated 550
in accordance with the highest ASIL, unless the software components meet the criteria for coexistence 551
in accordance with ISO 26262-9:2018, Clause 6. 552
7.4.9 If software partitioning (see Annex D) is used to implement freedom from interference between 553
software components it shall be ensured that: 554
a) the shared resources are used in such a way that freedom from interference of software partitions 555
is ensured; 556
NOTE 1 Tasks within a software partition are not free from interference among each other. 557
NOTE 2 One software partition cannot change the code or data of another software partition nor 558
command non-shared resources of other software partitions. 559
NOTE 3 The service received from shared re sources by one software partition cannot be affected by 560
another software partition. This includes the performance of the resources concerned, as well as the rate, 561
latency, jitter and duration of scheduled access to the resource. 562
b) the software partitioning is supported by dedicated hardware features or equivalent means (this 563
requirement applies to ASIL D, in accordance with 4.4); 564
EXAMPLE Hardware feature such as a memory protection unit. 565
c) the part of the software that implements the software partitioning is developed in compliance with 566
the highest ASIL assigned to any requirement of the software partitions; and 567
NOTE In general the operating system provides or supports software partitioning. 568
d) evidence for the effectiveness of the software partitioning is generated during software integration 569
and testing (in accordance with Clause 10). 570
7.4.10 Safety analysis shall be carried out at the software architectural level in accordance with 571
ISO 26262-9:2018, Clause 8, in order to: 572
 provide evidence for the suitability of the software to provide the specified safety-related functions 573
and properties with the integrity as required by the respective ASIL; 574
NOTE Safety-related properties include independency and freedom from interference requirements. 575
 identify or confirm the safety-related parts of the software; and 576
 support the specification and verify the effectiveness of the safety mechanisms. 577
NOTE 1 Safety mechanisms can be specified to cover both issues associated with random hardware failures as 578
well as software faults. 579
NOTE 2 See Annex E for additional information about the application of safety analyses at the software 580
architectural level. 581
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 24

ISO/DIS 26262-6:2016(E)
24 © ISO 2016 – All rights reserved

7.4.11 If the implementation of software safety requirements relies on freedom from interference or 582
sufficient independence between software components, dependent failures and their effects shall be 583
analysed in accordance with ISO 26262 -9:2018, Clause 7. 584
NOTE See Annex E and ISO 26262 -9:2018 Annex C for additional information about the application of 585
analyses of dependent failures at the software architectural level. 586
7.4.12 To specify the necessary software safety mechanisms at the software architectural level, based 587
on the results of the safety analysis at the software architectural level in accordance with 7.4.10 or 588
7.4.11, mechanisms for error detection as listed in Table 4 shall be applied. 589
Table 4 — Mechanisms for error detection at the software architectural level 590
Mechanisms ASIL
A B C D
1a Range checks of input and output data ++ ++ ++ ++
1b Plausibility check a
+ + + ++
1c Detection of data errors
b
+ ++ ++ ++
1d Monitoring of program execution
c
o + ++ ++
1e Temporal monitoring of program execution o + ++ ++
1f Diverse redundancy in the design o o + ++
1g Access permission control mechanisms d + + ++ ++
a Plausibility checks can include using a reference model of the desired behaviour, assertion checks, or comparing signals
from different sources.
b Types of mechanisms that may be used to detect data errors include error detecting codes and multiple data storage.
c An external element such as an ASIC or another software element performing a watchdog function. Monitoring can be
logical or temporal monitoring o r both.
d Control mechanisms may be implemented in software or hardware and are concerned with granting or denying access to
safety-related shared resources
7.4.13 This subclause applies to ASIL (A), (B), C and D, in accordance with 4.4: to specify the necessary 591
software safety mechanisms at the software architectural level, based on the results of the safety 592
analysis at software architectural level in accordance with 7.4.10 and 7.4.1 1, mechanisms for error 593
handling as listed in Table 5 shall be applied. 594
NOTE The analysis of possible hazards due to hardware is described in ISO 26262-5. 595
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 25

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 25

Table 5 — Mechanisms for error handling at the software architectural level 596
Mechanisms ASIL
A B C D
1a Static recovery mechanism a
+ + + +
1b Graceful degradation b
+ + ++ ++
1c Homogenous redundancy in the design c
+ + + ++
1d Diverse redundancy in the design d o o + ++
1e Correcting codes for data + + + +
1f Inhibit access permission violations e + + ++ ++
a Static recovery mechanisms can include the use of recovery blocks, backward recovery, forward recovery and recovery
through repetition.
b Graceful degradation at the software level refers to prioritizing functions to minimize the adverse effects of potential
failures on functional safety.
c Homogenous redundancy focuses primarily to control effects of transient faults or random faults in the hardware on
which a similar software is executed (e.g. temporal redundant exec ution of software ).
d
Diverse redundancy implies dissimilar software in each parallel path and focuses primarily on the prevention or control
of systematic faults in the software, common cause failures or common mode failures in the software or the hardware on
which the software is executed.
e
Access protection mechanisms may be implemented in software or hardware and are concerned with granting or
denying access to safety -related shared resources
7.4.14 An upper estimation of required resources for the embedded software shall be made, including: 597
a) the execution time; 598
b) the storage space; and 599
EXAMPLE RAM for stacks and heaps, ROM for program and non-volatile data. 600
c) the communication resources. 601
7.4.15 The software architectural design shall be verified in accordance with ISO 26262-8:2018, 602
Clause 9, and by using the software architectural design verification methods listed in Table 6 to 603
provide evidence that the following objectives are achieved: 604
a) the software architectural design is suitable to satisfy the software requirements with the required 605
integrity as stated by their ASIL ; 606
b) the analyses of the software architectural design provide evidence for the suitability of the design; 607
c) compatibility with the target h ardware; and 608
NOTE This includes the resources as specified in 7.4.1 4. 609
d) adherence to design guidelines. 610
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 26

ISO/DIS 26262-6:2016(E)
26 © ISO 2016 – All rights reserved

Table 6 — Methods for the verification of the software architectural design 611
Methods ASIL
A B C D
1a Walk-through of the design a
++ + o o
1b Inspection of the design a
+ ++ ++ ++
1c Simulation of dynamic parts of the design b
+ + + ++
1d Prototype generation o o + ++
1e Formal verification o o + +
1f Control flow analysis c
+ + ++ ++
1g Data flow analysis c
+ + ++ ++
1h Scheduling analysis + + ++ ++
a In the case of model-based development these methods can also be applied to the model.
b Method 1c requires the usage of executable models for the dynamic parts of the software architecture.
c Control and data flow analysis can be limited to safety -related components and their interfaces .
7.5 Work products 612
7.5.1 Software architectural design specification resulting from requirements 7.4.1 to 7.4.14. 613
7.5.2 Software safety requirements specification (refined) resulting from requirement 7.4. 6. 614
7.5.3 Safety analysis report resulting from requirement 7.4.1 0. 615
7.5.4 Dependent failures analysis report resulting from requirement 7.4. 11. 616
7.5.5 Software verification report (refined) resulting from requirement 7.4.1 5. 617
8 Software unit design and implementation 618
8.1 Objectives 619
The objectives of this subphase are: 620
a) to specify the software units in accordance with the software architectural design and the 621
associated software safety requirements; and 622
b) to implement the software units as specified . 623
8.2 General 624
Based on the software architectural design, the detailed design of the software units is developed. The 625
detailed design can be represented in the form of a model . The source code can be manually or 626
automatically generated from the design following coding or modelling guidelines. The functions and 627
properties are achievable at the source code level if manual code development is used. If model -based 628
development with automa tic code generation is used, these properties apply at both model and code 629
level, but can be verified only at model level provided that the code generation preserves these 630
properties. 631
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 27

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 27

In order to develop a single software unit design both software safety requirements as well as all non -632
safety-related requirements are implemented. Hence in this subphase safety -related and non -safety-633
related requirements are handled within one development process. 634
8.3 Inputs to this clause 635
8.3.1 Prerequisites 636
The following information shall be available: 637
 design and coding guidelines for modelling and programming languages in accordance with 5.5. 1; 638
 hardware -software interface specification (refined) in accordance with 6.5.2; 639
 software architectural design specification in accordance with 7.5.1; and 640
 software safety requirements specification (refined) in accordance with 7.5. 2. 641
8.3.2 Further supporting information 642
The following information can be considered: 643
 technical safety concept (see ISO 26262-4:2018, 7.5.1); 644
 system architectural design specification (see ISO 26262-4:2018, 7.5.2); 645
 tool application guidelines (see 5.5.2); 646
 safety analysis report (see 7.5.3); 647
 dependent failure analysis report (see 7.5.4); and 648
 guidelines for the application of methods (from an external source). 649
8.4 Requirements and recommendations 650
8.4.1 The requirements of this subclause shall be complied with if the software unit is safety-related. 651
NOTE “Safety-related” means that the unit implements safety requirements, or that the criteria for coexistence 652
(see ISO 26262-9:2018, Clause 6) of the unit with other units are not satisfied. 653
8.4.2 To avoid systematic faults and to ensure that the software unit design achieves the following 654
properties, the software unit design shall be described using the notations listed in Table 7. 655
a) unambiguity ; 656
b) consistency; 657
c) comprehensibility, ; 658
d) maintainability ; and 659
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 28

ISO/DIS 26262-6:2016(E)
28 © ISO 2016 – All rights reserved

e) verifiability. 660
Table 7 — Notations for software unit design 661
Notations ASIL
A B C D
1a Natural language a ++ ++ ++ ++
1b Informal notations ++ ++ + +
1c Semi-formal notations b + + ++ ++
1d Formal notations + + + +
a Natural language can complement the use of notations for example where some topics are more readily expressed in
natural language, or providing explanation and rationale for decisions captured in the notation.
EXAMPLE To avoid possible ambiguity of natural language when designing complex elements, a combination of an activity
diagram with natural language can be used.
b Semi-formal notations include pseudocode or modelling with UML®, SysML TM, Simulink® or Stateflow®, structured
sentences or requirement patterns with controlled vocabulary.
NOTE In the case of model -based development with automatic code generation, the methods for representing 662
the software unit design are applied to the model which serves as the basis for the code generation. 663
8.4.3 The specification of the software units shall describe the functional behaviour and the internal 664
design to the level of detail nece ssary for their implementation. 665
EXAMPLE Internal design can include constraints on the use of registers and storage of data. 666
8.4.4 Design principles for software unit design and implementation at the source code level as listed 667
in Table 8 shall be applied to achieve the following properties: 668
a) correct order of execution of subprograms and functions within the software units, based on the 669
software architectural design; 670
b) consistency of the interfaces between the software units; 671
c) correctness of data flow and control flow between and within the software units; 672
d) simplicity; 673
e) readability and comprehensibility; 674
f) robustness; 675
EXAMPLE Methods to prevent implausible values, execution errors, division by zero, and errors in the data 676
flow and control flow. 677
g) suitability for software modification; and 678
h) verifiability. 679
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 29

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 29

Table 8 — Design principles for software unit design and implementation 680
Principle ASIL
A B C D
1a One entry and one exit point in subprograms and functions a
++ ++ ++ ++
1b No dynamic objects or variables, or else online test during their creation a, b
+ ++ ++ ++
1c Initialization of variables ++ ++ ++ ++
1d No multiple use of variable names a
++ ++ ++ ++
1e Avoid global variables or else justify their usage a
++ ++ ++ ++
1f Limited use of pointers a
+ ++ ++ ++
1g No implicit type conversions a, b
+ ++ ++ ++
1h No hidden data flow or control flow c
+ ++ ++ ++
1i No unconditional jumps a, b, c
++ ++ ++ ++
1j No recursions + + ++ ++
a Principle 1a, 1b, 1d, 1e, 1f, 1g and 1i may not be applicable for graphical modelling notations used in model -based
development.
c Principle 1h and 1i reduce the potential for modelling data flow and control flow through jumps or global variables.
NOTE For the C language, MISRA C[3] covers many of the principles listed in Table 8. 681
8.5 Work products 682
8.5.1 Software unit design specification resulting from requirements 8.4.2 to 8.4.4. 683
NOTE In the case of model -based development, the implementation model and supporting descriptive 684
documentation, using methods listed in Table 8, specifies the software units. 685
8.5.2 Software unit implementation resulting from requirement 8.4.4. 686
9 Software unit verification 687
9.1 Objectives 688
The objective of this subphase is to provide evidence that the software detailed design and the 689
implemented software units fulfil the ir requirements and do not contain undesired f unctionality. 690
9.2 General 691
The software detailed design and the implemented software units are verified by using an appropriate 692
combination of verification measures such as reviews, analyses and testing. 693
9.3 Inputs to this clause 694
9.3.1 Prerequisites 695
The following information shall be available: 696
 hardware -software interface (HSI) specification (refined) in accordance with 6.5.2; 697
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 30

ISO/DIS 26262-6:2016(E)
30 © ISO 2016 – All rights reserved

 software unit design specification in accordance with 8.5.1; and 698
 software unit implementation in accordance with 8.5.2 . 699
9.3.2 Further supporting information 700
The following information can be considered: 701
 tool application guidelin es (see 5.5.2); and 702
 guidelines for the application of methods (from an external source). 703
9.4 Requirements and recommendations 704
9.4.1 The requirements of this subclause shall be complied with if the software unit is safety-related. 705
NOTE “Safety-related” means that the unit implements safety requirements, or that the criteria for coexistence 706
of the unit with other units are not satisfied. 707
9.4.2 Software unit verification shall be planned, specified and executed in accordance with 708
ISO 26262-8:2018, Clause 9. 709
NOTE 1 Based on the definitions in ISO 26262-8:2018, Clause 9, the verification objects in the software unit 710
verification are the software units. 711
NOTE 2 For model-based software development, the corresponding parts of the implementation model also 712
represent objects for the verification planning. Depending on the selected software development process the 713
verification objects can be the code derived from this model or the model itself. 714
9.4.3 The software unit design and the implemented software unit shall be verified by an appropriate 715
combination of methods according to Table 9 to provide evidence for: 716
a) compliance with the software unit design specification (in accordance with Clause 8); 717
b) compliance with the specification of the hardware-software interface (in accordance with Clause 6), 718
if applicable; 719
c) compliance with the software safety requirements allocated to the softw are units (in accordance 720
with 7.4. 6); 721
d) confidence in the absence of unintended functionality and properties ; 722
e) robustness; and 723
EXAMPLE The effectiveness of error detection and error handling mechanisms in order achieve 724
robustness of the software unit. 725
f) sufficient resources to support their functionality and properties . 726
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 31

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 31

Table 9 — Methods for software unit verification 727
Methods ASIL
A B C D
1a Walk-through a
++ + o o
1b Pair-programming + + + +
1c Inspection a
+ ++ ++ ++
1d Semi-formal verification + + ++ ++
1e Formal verification o o + +
1f Control flow analysis b, c
+ + ++ ++
1g Data flow analysis b, c
+ + ++ ++
1h Static code analysis
d
++ ++ ++ ++
1i Static analyses based on abstract interpretation
e
+ + + ++
1j Requirements-based test
f
++ ++ ++ ++
1k Interface test ++ ++ ++ ++
1l Fault injection test g
+ + + ++
1m Resource usage test h + + + ++
1n Back-to-back comparison test between model and code, if applicable i + + ++ ++
a
For model -based development these methods are applied at the model level, if evidence is available that justifies
confidence in the used code generator .
b
Methods 1 f and 1 g can be applied at the source code level. These methods are applicable both to manual code
development and to model -based development.
c
Methods 1f and 1g can be part of methods 1 e, 1h or 1i.
d
Static analyses include MISRA rule checks and analyses of reso urce consumption.
e
Method 1 i is used for mathematical analysis of source code by use of an abstract representation of possible values for the
variables. For this it is not necessary to translate and execute the source code.
f The software requirements at the unit level are the basis for this requirements -based test.
g In the context of software unit testing, fault injection test means to means to modify the tested software unit (e.g.
introduce faults into the software) to test the robustness. Such modifica tions include injection of arbitrary faults (e.g. by
corrupting values of variables, by introducing code mutations, or by corrupting values of CPU registers).
h Some aspects of the resource usage test can only be evaluated properly when the software unit tests are executed on the
target hardware or if the emulator for the target processor supports resource usage tests.
i This method requires a model that can simulate the functionality of the software units. Here, the model and code are
stimulated in the sa me way and results compared with each other.
EXAMPLE In the case of model-based design results of non -floating-point operations can be compared .
9.4.4 To enable the specification of appropriate test cases for the software unit testing in accordance 728
with 9.4.3, test cases shall be derived using the methods as listed in Table 10. 729
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 32

ISO/DIS 26262-6:2016(E)
32 © ISO 2016 – All rights reserved

Table 10 — Methods for deriving test cases for software unit testing 730
Methods ASIL
A B C D
1a Analysis of requirements ++ ++ ++ ++
1b Generation and analysis of equivalence classes a
+ ++ ++ ++
1c Analysis of boundary values b
+ ++ ++ ++
1d Error guessing c
+ + + +
a Equivalence classes can be identified based on the division of inputs and outputs, such that a representative test value
can be selected for each class.
b This method applies to interfaces, values approaching and crossing the boundaries and out of range values.
c Error guessing tests can be based on data collected through a “lessons learned” process and expert judgment.
9.4.5 To evaluate the completeness of verification and to provide evidence that the test objectives for 731
unit testing are adequately achieved , the coverage of requirements at the software unit level shall be 732
determined and the structural coverage shall be measured in accordance with the metrics as listed in 733
Table 11. If the achieved structural coverage is considered insufficient, either additional test cases shall 734
be specified or a rationale based on other methods shall be provided. 735
EXAMPLE 1 Analysis of structural coverage can reveal shortcomings in requirement -based test cases, 736
inadequacies in requirements, dead code, deactivated code or unintended functionality. 737
EXAMPLE 2 A rationale can be given for the level of coverage achieved based on accepted dead code (e.g. code 738
for debugging) or code segments depending on different software configurations; or code not covered can be 739
verified using complementary methods (e.g. inspections). 740
EXAMPLE 3 A rationale can be based on state of the art and technology. 741
EXAMPLE 4 No target value or low target value for structural coverage without a rationale is considered 742
insufficient. 743
Table 11 — Structural coverage metrics at the software unit level 744
Methods ASIL
A B C D
1a Statement coverage ++ ++ + +
1b Branch coverage + ++ ++ ++
1c MC/DC (Modified Condition/Decision Coverage) + + + ++
NOTE 1 The structural coverage can be determined by the use of appropriate software tools. 745
NOTE 2 In the case of model -based development, the analysis of structural coverage can be performed at the 746
model level using analogous structural coverage metrics for models. 747
EXAMPLE 5 The analysis of structural coverage performed at the model level may replace the source code 748
coverage metrics if it is shown to be equivalent with rationales based on evidence that the coverage is 749
representative of the code level. 750
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 33

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 33

NOTE 3 If instrumented code is used to determine the degree of coverage, it can be necessary to provide evidence 751
that the instrumentation has no effect on the test results. This can be done by repeating the tests with non -752
instrumented code. 753
9.4.6 The test environment for software unit testing shall correspond as closely as possible to the 754
target environment. If the software unit testing is not carried out in the target environment, the 755
differences in the source and object code, and the differ ences between the test environment and the 756
target environment, shall be analysed in order to specify additional tests in the target environment 757
during the subsequent test phases. 758
NOTE 1 Differences between the test environment and the target environment ca n arise in the source code or 759
object code, for example, due to different bit widths of data words and address words of the processors. 760
NOTE 2 Depending on the scope of the tests, the appropriate test environment for the execution of the software 761
unit is used (e.g. the target processor, a processor emulator or a development system). 762
NOTE 3 Software unit testing can be executed in different environments, for example: 763
 model-in-the-loop tests, 764
 software-in-the-loop tests, 765
 processor-in-the-loop tests, or 766
 hardware-in-the-loop tests. 767
NOTE 4 For model-based development, software unit testing can be carried out at the model level followed by 768
back-to-back comparison tests between the model and the object code. The back -to-back comparison tests are 769
used to ensure that t he behaviour of the models with regard to the test objectives is equivalent to the 770
automatically-generated code. 771
9.5 Work products 772
9.5.1 Software verification specification resulting from requirements 9.4.2 to 9.4.6. 773
9.5.2 Software verification report (refined) resulting from requirement 9.4.2. 774
10 Software integration and verification 775
10.1 Objectives 776
The objectives of this subphase are: 777
a) to integrate the software elements; and 778
b) to provide evidence that the embedded software complies with the software architectural design. 779
10.2 General 780
In this subphase, the particular integration levels and the interfaces between the software elements are 781
verified against the software architectural design. The steps of the integration and verification of the 782
software elements correspond directly to the hierarchical architecture of the software. 783
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 34

ISO/DIS 26262-6:2016(E)
34 © ISO 2016 – All rights reserved

The embedded software can consist of safety -related and non -safety-related software elements. 784
10.3 Inputs to this clause 785
10.3.1 Prerequisites 786
The following information shall be av ailable: 787
 hardware -software interface (HSI) specification (refined) in accordance with 6.5.2; 788
 software architectural design specification in accordance with 7.5.1; 789
 software unit implementation in accordance with 8.5.2; and 790
 software verification specification in accordance with 9.5. 1. 791
10.3.2 Further supporting information 792
The following information can be considered: 793
 qualified software components available (see ISO 26262-8:2018, Clause 12); 794
 software tool qualification report (see ISO 26262-8:2018, 11.5.2); 795
 tool application guidelin es (see 5.5.2); and 796
 guidelines for the application of methods (from an external source). 797
10.4 Requirements and recommendations 798
10.4.1 The planning of the software integration shall describe the steps for integrating the individual 799
software units hierarchically into software components until the embedded software is fully integrated, 800
and shall consider: 801
a) the functional dependencies that are relevant for software integration; and 802
b) the dependenci es between the software integration and the hardware -software integration. 803
NOTE For model-based development, the software integration can be replaced with integration at the model 804
level and subsequent automatic code generation from the integrated model. 805
10.4.2 Software integration verification shall be planned, specified and executed in accordance with 806
ISO 26262-8:2018, Clause 9. 807
NOTE 1 Based on the definitions in ISO 26262-8:2018, Clause 9, the software integration verification objects 808
are the software components. 809
NOTE 2 For model-based development, the verification objects can be the models associated with the software 810
components. 811
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 35

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 35

10.4.3 The software integration shall be verified by an appropriate combination of methods according 812
to Table 1 2 to provide evidence that both the software components and the embedded software 813
achieve: 814
a) compliance with the software architectural design in accordance with Clause 7; 815
b) compliance with the hardware -software interface specification in accordance with 6. 5.2; 816
c) the specified functionality; 817
d) the specified properties ; 818
EXAMPLE Reliability due to absence of inaccessible software, robustness against erroneous inputs, 819
dependability due to effective error detection and handling. 820
e) sufficient resources to support the func tionality; and 821
f) effectiveness of the software partitioning , if applicable. 822
Table 12 — Methods for verification of software integration 823
Methods ASIL
A B C D
1a Requirements-based test
a
++ ++ ++ ++
1b Interface test ++ ++ ++ ++
1c Fault injection test b
+ + ++ ++
1d Resource usage test c, d
++ ++ ++ ++
1e Back-to-back comparison test between model and code, if applicable e
+ + ++ ++
1f Analyses of the control or data flow + + ++ ++
1g Static code analysis
f
++ ++ ++ ++
1h Static analyses based on abstract interpretation g + + + +
a The software requirements at the architectural level are the basis for this requirements -based test.
b In the context of software integration testing, fault injection test means to introduce faults into the software for the
purposes described in Clause 10.4.3 and in particular to test the correctness of hardware -software interface related to safety
mechanisms. This includes injection of arbitrar y faults in order to test safety mechanisms (e.g. by corrupting software
interfaces). Fault injection can also be used to verify freedom from interference.
c To ensure the fulfilment of requirements influenced by the hardware architectural design with suf ficient tolerance,
properties such as average and maximum processor performance, minimum or maximum execution times, storage usage (e.g.
RAM for stack and heap, ROM for program and data) and the bandwidth of communication links (e.g. data buses) have to be
determined.
d Some aspects of the resource usage test can only be evaluated properly when the software integration tests are executed
on the target hardware or if the emulator for the target processor supports resource usage tests.
e This method requires a model that can simulate the functionality of the software components. Here, the model and code
are stimulated in the same way and results compared with each other.
f Static analyses include modelling or coding rule checks (e.g. MISRA) and analyses of res ource consumption.
g Abstract interpretation of the source code, analyses of pointers, invalid uses of locks.
NOTE 2 For model-based development, the verification objects can be the models associated with the software 824
components. 825
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 36

ISO/DIS 26262-6:2016(E)
36 © ISO 2016 – All rights reserved

10.4.4 To enable the spec ification of appropriate test cases for the software integration test methods 826
selected in accordance with 10.4.3, test cases shall be derived using the methods as listed in Table 13. 827
Table 13 — Methods for deriving test cases for software integration testing 828
Methods ASIL
A B C D
1a Analysis of requirements ++ ++ ++ ++
1b Generation and analysis of equivalence classes a + ++ ++ ++
1c Analysis of boundary values b
+ ++ ++ ++
1d Error guessing c
+ + + +
a
Equivalence classes can be identified based on the division of inputs and outputs, such that a representative test value
can be selected for each class.
b
This method applies to parameters or variables, values approaching and crossing the boundaries and out of range values.
c
Error guessing tests can be based on data collected through a “lessons learned” process and expert judgment.
10.4.5 To evaluate the completeness of verification and to provide evidence that the test objectives for 829
integration testing are adequately achieved, the coverage of requirements at the software architectural 830
level by test cases shall be determined. If necessary, additional test cases shall be specified or a 831
rationale based on other methods shall be provided. 832
10.4.6 This subclause applies to ASIL (A), (B), C and D, in accordance with 4. 4: To evaluate the 833
completeness of test cases and to provide evidence that the test objectives for integration testing are 834
adequately achieved , the structural coverage shall be evaluated in accordance with the methods as 835
listed in Table 14. If the achieved structural coverage is considered insufficient, either additional test 836
cases shall be specified or a rationale based on other methods shall be provided. 837
EXAMPLE Analysis of structural coverage can reveal shortcomings in the requirement -based test cases, 838
inadequacies in the requirements, dead code, deactivated code or unintended functionality. 839
Table 14 — Structural coverage at the software architectural level 840
Methods ASIL
A B C D
1a Function coverage a
+ + ++ ++
1b Call coverage b
+ + ++ ++
a Method 1a refers to the percentage of executed software functions. This evidence can be achieved by an appropriate
software integration strategy.
b Method 1b refers to the percentage of executed software function calls.
NOTE 1 The structural coverage can be determined using appropriate software tools. 841
NOTE 2 In the case of mo del-based development, software architecture testing can be performed at the model 842
level using analogous structural coverage metrics for models. 843
10.4.7 It shall be verified that the embedded software that is to be included as part of a production 844
release in accordance with ISO 26262-2:2018, Clause 6.4.10, contains all the specified functions and 845
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 37

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 37

properties, and only contains other unspecified functions if these functions do not impair the 846
compliance with the software safety requirements. 847
EXAMPLE In this context unspecified functions include code used for debugging or instrumentation. 848
NOTE If deactivation of these unspecified functions can be ensured, this is an acceptable means of compliance 849
with this requirement. Otherwise the removal of such code is a change (see ISO 26262-8:2018, Clause 8). 850
10.4.8 The test environment for software integration testing shall correspond as closely as possible to 851
the target environment. If the software integration testing is not carried out in the target environment, 852
the differences in the source and object code and the differences between the test environment and the 853
target environment shall be analysed in order to specify additional tests in the target environment 854
during the subsequent test phases. 855
NOTE 1 Differences between the test environment and the target environment can arise in the source or object 856
code, for example, due to different bit widths of data words and address words of the processors. 857
NOTE 2 Depending on the scope of the tests and the hierarchical level of integration, the appropriate test 858
environments for the execution of the software elements are used. Such test environments can be the target 859
processor for final integration, or a processor emulator or a development system for the previous integration 860
steps. 861
NOTE 3 Software integration testing can be executed in different environments, for example: 862
 model-in-the-loop tests, 863
 software-in-the-loop tests, 864
 processor-in-the-loop tests, or 865
 hardware-in-the-loop tests. 866
10.5 Work products 867
10.5.1 Software verification specification (refined) resulting from requirements 10.4.1 to 10.4.8. 868
10.5.2 Embedded software resulting from requirement 10.4.1. 869
10.5.3 Software verification report (refined) resulting from requirement 10.4.2. 870
11 Testing of the embedded software 871
11.1 Objective 872
The objective of this subphase is to provide evidence that the embedded software fulfils the software 873
safety requirements. 874
11.2 General 875
The purpose of this activity is to provide evidence that the embedded software fulfils its requirements 876
in the target environment. 877
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 38

ISO/DIS 26262-6:2016(E)
38 © ISO 2016 – All rights reserved

11.3 Inputs to this clause 878
11.3.1 Prerequisites 879
The following information shall be available: 880
 software architectural design specification in accordance with 7.5.1; 881
 software safety requirements specification (refined) in accordance with 7.5.2; and 882
 software verification specification (refined) in accordance with 10.5. 1. 883
11.3.2 Further supporting information 884
The following information can be considered: 885
 technical safety concept (see ISO 26262-4:2018, 7.5.1); 886
 system architectural design specification (see ISO 26262-4:2018, 7.5.2); 887
 integration and verification report (see ISO 26262-4:2018, 8.5. 2); 888
 tool application guidelines (see 5.5.2); and 889
 guidelines for the application of methods (from an external source). 890
11.4 Requirements and recommendati ons 891
11.4.1 To verify that the embedded software fulfils the software safety requirements, tests shall be 892
conducted in the test environments as listed in Table 15. 893
NOTE Test cases that already exist, for example from software integration testing, can be re-used. 894
Table 15 — Test environments for conducting the software testing 895
Methods ASIL
A B C D
1a Hardware-in-the-loop ++ ++ ++ ++
1b Electronic control unit network environments a
++ ++ ++ ++
1c Vehicles + + ++ ++
a Examples include test benches partially or fully integrating the electrical systems of a vehicle, “lab -cars” or “mule”
vehicles, and “rest of the bus” simulations.
896
11.4.2 The software test cases derived by methods as listed in Table 16 shall be executed to provide 897
evidence that the embedded software fulfils the software requirements as required by their respective 898
ASIL. 899
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 39

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 39

Table 16 — Methods for deriving test cases for the test of the embedded software 900
Methods ASIL
A B C D
1a Analysis of requirements ++ ++ ++ ++
1b Generation and analysis of equivalence classes + ++ ++ ++
1c Analysis of boundary values + + ++ ++
1d Error guessing based on knowledge or experience + + ++ ++
1e Analysis of functional dependencies + + ++ ++
1f Analysis of operational use cases + ++ ++ ++
1g Fault injection test
a
+ + + ++
a In the context of software testing, fault injection test means to introduce faults into the software by means of e.g.
corrupting calibration parameters.
901
11.4.3 The results of the verification of the software safety requirements shall be evaluated with 902
regard to: 903
a) compliance with the expected results; 904
b) coverage of the software safety requirements; and 905
NOTE this includes the coverage of the configuration and calibration ranges. See Annex C. 906
c) pass or fail criteria. 907
11.5 Work products 908
11.5.1 Software verification specification (refined) resulting from requirements 11.4.1 to 11.4.3. 909
11.5.2 Software verification report (refined) resulting from requirements 11.4.1 to 11.4.3. 910
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 40

ISO/DIS 26262-6:2016(E)
40 © ISO 2016 – All rights reserved

Annex A (informative) Overview of and workflow of management of product 911
development at the software l evel 912
Table A.1 provides an overview of objectives, prerequisites and work products of the particular phases 913
of the product development at the software level. 914
Table 17 — Overview of product development at the software level 915
Clause Objectives Prerequisites Work products
5
General topics
for the
product
development
at the softwar e
level
The objective of this clause is
to give an overview of the
product development at the
software level and to
provide topics specific to
software development.
(none) 5.5.1 Design and coding
guidelines for modelling and
programming languages
5.5.2 Tool application
guidelines
6
Specification
of software
safety
requirements
The objectives of this
subphase are:
a) to specify the software
safety requirements.
They are derived from
the technical safety
concept and the system
design specification;
b) to refine the hardware -
software interface
requirements initiated
in ISO 26262 -4:2018,
Clause 7; and
c) to verify that the
software safety
requirements and the
hardware-software
interface requirements
are consistent with the
technical safety concept
and the system design
specification.
Technical safety concept
(see ISO 26262-4:2018, 7.5.1)
System architectural design
specification
(see ISO 26262-4:2018, 7.5.2)
Hardware-software interface
(HSI) specification
(see ISO 26262-4:2018, 7.5.6)
6.5.1 Software safety
requirements specification
6.5.2 Hardware-software
interface (HSI) specification
(refined)
6.5.3 Software verification
report
7
Software
architectural
design
The objective of this
subphase are:
a) to develop a software
architectural design that
satisfies the software
safety requirements and
the non -safety-related
requirements;
b) to verify that the
software architectural
design is suitable to
satisfy the software
safety requirements
with the required ASIL;
and
c) to support the
implementation of the
software.
Design and coding guidelines
for modelling and programming
languages (see 5.5.1)
Hardware-software interface
(HSI) specification (refined)
(see 6.5.2)
Software safety requirements
specification (see 6.5.1)
7.5.1 Software architectural
design specification
7.5.2 Software safety
requirements specification
(refined)
7.5.3 Safety analysis report
7.5.4 Dependent failures
analysis report
7.5.5 Software verification
report (refined)
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 41

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 41

916
Table A.1 (continued) 917
Clause Objectives Prerequisites Work products
8
Software
unit design
and
implementation
The objectives of this
subphase are:
a) to specify the software
units in accordance with
the software architectural
design and the associated
software safety
requirements; and
b) to implement the softwar e
units as specified.
Design and coding guidelines
for modelling and programming
languages (see 5.5.1)
Hardware-software interface
(HSI) specification (refined)
(see 6.5.2)
Software architectural design
specification (see 7.5.1)
Software safety requirements
specification (refined) (see
7.5.2)
8.5.1 Software unit design
specification
8.5.2 Software unit
implementation
9
Software
unit
verification
The objective of this subphase
is to provide evidence that the
software detailed design and
the implemented software
units fulfil their requirements
and do not contain undesired
functionality.
Hardware-software interface
(HSI) specification (refined)
(see 6.5.2)
Software unit design
specification (see 8.5.1)
Software unit implementation
(see 8.5.2)
9.5.1 Software verification
specification
9.5.2 Software verification
report (refined)
10
Software
integration
and
verification
The objectives of this
subphase are:
a) to integrate the software
elements; and
b) to provide evidence that
the embedded software
complies with the
software architectural
design.
Hardware-software interface
(HSI) specification (refined)
(6.5.2)
Software architectural design
specification (see 7.5.1)
Software unit implementation
(see 8.5.2)
Software verification
specification (see 9.5.1)
Software verification report
(refined) (see 9.5.2)
10.5.1 Software verification
specification (refined)
10.5.2 Embedded software
10.5.3 Software verification
report (refined)
11
Testing of
the
embedded
software
The objective of this subphase
is to provide evidence that the
embedded software fulfils the
software safety requirements.
Software architectural design
specification (see 7.5.1)
Software safety requirements
specification (refined) (see
7.5.2)
Software verification
specification (refined) (see
10.5.2)
11.5.1 Software verification
specification (refined)
11.5.2 Software verification
report (refined)
Annex C
Software
configuration
The objective of software
configuration is to enable
controlled changes in the
behaviour of the software for
different applications
See applicable prerequisites of
the relevant phases of the safet y
lifecycle in which software
configuration is applied.
C.5.1 Configuration data
specification
C.5.2 Calibration data
specification
C.5.3 Configuration data
C.5.4 Calibration data
C.5.5 Software verification plan
(refined)
C.5.6 Verification specification
(refined)
C.5.7 Verification report
(refined)
918
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 42

ISO/DIS 26262-6:2016(E)
42 © ISO 2016 – All rights reserved

Annex B (informative) Model-based development approaches 919
B.1 Objectives 920
This annex explains possible usage benefits and risks of model-based development approaches (MBDV) 921
during the development at the software level. 922
NOTE 1 Risks mentioned in this annex can potentially cause the violation of safety requirements and safety 923
goals. 924
NOTE 2 This annex does not imply that the model-based development approaches mentioned are restricted to 925
software development only. 926
B.2 General 927
This sectio n provides general information that is common to various use cases of MBDV (e.g. 928
requirements, design, verification/testing). Section B.3 provides specific considerations based on 929
exemplary use cases. 930
B.2.1 Introduction 931
Models may be used to represent information or views of information required during the development 932
phase as well as for the abstract design and implementation of one or more software elements or the 933
software’s environment. Models themselves may consist of various hierarchical refinement le vels or 934
references to refined models at a lower hierarchical level (e.g. hierarchical structure with black box and 935
white box views for each model element). A model is developed using a model ling notation. 936
Modelling notations may be graphical and/or textua l. They may be formal (e.g. notation with an 937
underlying mathematical definition) or semi-formal (e.g. structured notation with incompletely defined 938
semantics). Modelling notations may be international standards (e.g. UML) or company -specific. They 939
are typi cally based on concepts such as classes, block diagrams, control diagrams, or state charts 940
(expressing states and transition between states). In this annex, only models and modelling notations 941
are considered which incorporate a defined syntax and semantics adequate for their intended use (i.e. 942
illustrative figures without such definitions are not considered as models). Beside the specific reasons 943
for using MBDV (e.g. simulation or code generation) , an adequately defined syntax and semantic s is a 944
basis for the achievement of criteria such as comprehensibility, unambiguity, correctness, consistency 945
and verifiability of information or work products described by models especially when different parties 946
are collaborating. 947
In addition to the model ling notation i tself, model ling guidelines and/or suitable tools are means to 948
achieve adequately defined syntax and semantic s. 949
EXAMPLE As part of guidelines naming conventions or style guides support the achievement of criteria such 950
as “comprehensibility” while the selection of language subsets excluding ambiguous constructs supports both 951
“comprehensibility” and the correct transformation and execution of the model. In case of simulation or 952
transformation of models suitable simulation engines or code generators can supplement the required semantics 953
such as a deterministic order of execution or transformation. 954
In this annex the following use cases of model-based development approaches are considered further: 955
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 43

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 43

 Specification of software safety requirements (C lause 6 and ISO 26262 -8:2018, Clause 6) ; 956
 Development of the software architectural design (Clause 7) ; 957
 Design and implementation of software units, with or without automated code generation 958
(Clause 8); 959
 Design, implementation and integration of software comp onents, incl uding automated code 960
generation from software component models (Clauses 8, 9 and 10); and 961
 Verification (static and/or dynamic ) (Clauses 9, 10 and 11) . 962
B.2.2 General suitability aspects 963
The suitability of the selected model -based development app roach for an intended application, 964
including the modelling techniques, notations, languages or tools, can be evaluated based on aspects 965
such as: 966
 achievement of criteria provided by ISO 26262 for the corresponding development phase (e.g. 967
comprehensibility, completeness, verifiability, unambiguity, accuracy) with the selected 968
approach(es), 969
 knowledge and experiences with the selected approach(es) , 970
 adequacy of the definition of syntax and semantics , 971
 adequacy for the application domain (e.g. real -time behavio ur, data structures, generation of 972
software for a fixed-point versus floating-point microcontroller), 973
 role of the model in the software lifecycle (e.g. development of safety requirements, expressing 974
static or dynamic aspects of architectural designs) , 975
 support for managing complexity by enforcing the principles of abstraction, hierarchy and 976
modularity, 977
 collaboration model between internal/external development partners (e.g. consistency of data 978
during data exchange) , 979
 required versus available confidence in software tools (see ISO 26262-8:2018, Clause 11) , or 980
 qualification of software components (e.g. software libraries) used as part of selected approach or 981
included into selected tools (see ISO 26262 -8:2018, Clause 12) . 982
B.2.3 Potential impact of MBDV on the software lifecycle 983
A model may represent more than one work product of ISO 26262 -6 (e.g. requirements, architectural 984
design, detailed design and model-based software integration with code generation). In comparison to a 985
traditional development process where lifecycle data are separated, a stronger coalescence of the 986
phases “Software safety requirements”, “Software architectural design”, “Software unit design and 987
implementation” etc. may occur. The potential benefits of this approach (e.g. continuity, information 988
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 44

ISO/DIS 26262-6:2016(E)
44 © ISO 2016 – All rights reserved

sharing across the software life cycle, consistency) are appealing, but this approach may also introduce 989
risks (see clause B.3). 990
B.3 Specific considerations for the example software development use cases 991
This section provides considerations related to benefits and risks based on example use cases. 992
B.3.1 Specification of software safety requirements (Clause 6 and ISO 26262 -8:2018, 993
Clause 6) 994
Models can be used to capture the functionality, behaviour, design or properties of the software to be 995
realized at safety requirements level (e.g. open/closed loop controls, controlled systems, states and 996
transitions, monitoring functions or independence properties) and to enable verification and validation 997
of requirements by simulation or formal methods. 998
Models are a suitable mean s to achieve the characteristics of safety requirements and to satisfy 999
requirements to use semi-formal or formal notations as stated in ISO 26262-8, Clause 6. 1000
EXAMPLE When specifying a state machin e using a model based approach, the corresponding model 1001
captures the required states and transitions in an intuitive, unambiguous and comprehensible way. The fact that 1002
such a model represents more than one atomic requirement compared with a textual specification of the same 1003
state machine is acceptable. An adequate rationale would be that for this purpose completeness, verifiability and 1004
internal consistency of the specification can be achieved and maintained better by the model without duplication 1005
of information. 1006
However be aware that by using requirement models it is possible to miss safety requirements that 1007
cannot be expressed by the selected model-based development approach. For this reason ISO 26262-8, 1008
Clause 6 emphasizes an appropriate combination of tex tual requirements and other notations. 1009
EXAMPLE Requirements related to the robustness of the software functionality against missing or erroneous 1010
inputs 1011
B.3.2 Development of the software architectural design (Clause 7) 1012
Models can be used to capture the soft ware architecture according to several perspectives, in 1013
particular: 1014
1015
a) Static aspects (e.g. components, their interfaces and data flows, components attributes) ; and 1016
b) Dynamic aspects (e.g. functional chain of events, scheduling, message passing). 1017
Models together with the modelling guidelines are a suitable mean to achieve the properties of and 1018
principles for safety-related architectural designs and to satisfy requirements for semi-formal or formal 1019
notations as stated in Clause 7. 1020
However be aware that one modelling approach might not express all of the required aspects (e.g. 1021
models targeting the implementation of software functions often do not describe the overall 1022
architectural design including elements of the basic software or operating system) and that a model 1023
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 45

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 45

may not be fully understandable without further textual explanations (e.g. by different collaborating 1024
parties). 1025
B.3.3 Design and implementation of software units, with or without automated code 1026
generation (Clauses 8 and 9) 1027
Models together with the modelling and coding guidelines can be used to develop the detailed design 1028
and the software unit itself at a higher level of abstraction. Depending on the selected approach the 1029
software unit can be implemented by code generation from the model. 1030
This approach enables automated consistency checks (e.g. data typing, correct definition of data before 1031
use) and verification by formal methods, simulation, static analyses or model-in-the-loop/software-in-1032
the-loop testing of the software units. Depending o n the confidence in the code generator automated 1033
code generation from the model may lower the risk of coding errors compared to manual coding. 1034
However be aware that for example: 1035
 Design aspects that cannot be expressed by the model ling notations might be missed, if not 1036
documented in some other appropriate form. Even unit testing (e.g. using back -to-back-testing) 1037
might not reveal this, if all tests are only derived from such an “incomplete” model. 1038
 Safety-related code properties such as robustness might not be implemented, if only the intended 1039
function is modelled. Code properties often are achieved by an appropriate combination of the 1040
model properties and code generation properties (e.g. if the division is generated “as is” by the code 1041
generator, then avoidance of division by zero will be checked at model level. If the code generator 1042
handles the division by zero with a robust design coding pattern, division by zero analysis in the 1043
model may be alleviated). 1044
 Incorrect addressing or handl ing of numeric accuracy aspects might introduce faults (e.g. if the 1045
model is just inherited from control engineering). For instance issues resulting from different 1046
precision properties of the control engineer ’s computer run time environment and the target 1047
microcontroller run time environment. 1048
 The creation of discrete models of the software behaviour from continuous models may introduce 1049
systematic faults (e.g. effect of discretization of values and/or time, execution time of the 1050
calculation, handling concurr ent tasks). 1051
 Simulation or verification results (e.g. results of consistency checks) may be tool -dependent 1052
because different simulation engines or model checkers may deviate regarding the implemented 1053
semantics (e.g. for algorithmic aspects not addressed by the semantic s of the modelling language). 1054
 The resource usage in case of auto -coding (execution time and memory consumption) may be 1055
higher than in case of manual coding. 1056
 In case of auto-coding verifiability of the generated source code equivalence to manual code may be 1057
more difficult to achieve. 1058
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 46

ISO/DIS 26262-6:2016(E)
46 © ISO 2016 – All rights reserved

B.3.4 Design, implementation and integration of software components, including 1059
automated code generation from software component models (Clauses 8, 9 and 10) 1060
Models together with the modelling and coding gu idelines can be used to design, implement and 1061
integrate software at a higher level of abstraction. Thus software components or even the embedded 1062
software can be implemented by code generation from the integrated model. 1063
This approach implies a strong coalescence of the different software development phases and therefore 1064
the benefits versus risks of such an approach need to be evaluated carefully (see hints provided above) 1065
including the implementation of appropriate safety measures (e.g. using separate test m odels for 1066
deriving additional test cases). 1067
B.3.5 Verification (static and/or dynamic) (Clauses 9, 10 and 11) 1068
Models can serve as a mean for verifying work products (e.g. design model, code). Such a model is called 1069
a “verification model”. Verification models can be used for example as the basis for back-to-back testing, 1070
generation of test cases or for expressing safety-related properties as a reference for formal verification 1071
(this requires that the model to verify is formal too). 1072
Models can also serve as a means to enable or support verification activities (e.g. plant model needed 1073
for the hardware -in-the-loop testing of closed -loop controls by simulating the environment of the 1074
device under test). 1075
The use of model s for verification purposes may enable the e arly and efficient detection of 1076
faults/failure s in work products or software, more efficient test case generation, highly automated 1077
testing or even formal verification techniques. 1078
However be aware that for example: 1079
 Without evidence for the suitability of a model used for verification purposes the validity of related 1080
results might be questionable (e.g. inadequate plant models may lead to incorrect test results) . 1081
 For valid results the maturity of the verification model must match the target maturity of the device 1082
under test to be verified. 1083
 Test cases generated from the same model which is also used for code generation cannot serve as 1084
the only source for verifying both the model itself and the code generated from it. 1085
Annex C (normative) Software configuration 1086
C.1 Objectives 1087
The objective of software configuration is to enable controlled changes in the behaviour of the software 1088
for different applications. 1089
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 47

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 47

C.2 General 1090
Configurable software enables the development of application specific software using configuration and 1091
calibration data (see Figure C.1). 1092
1093
Figure 2 — Creating application specific software 1094
C.3 Inputs to this clause 1095
C.3.1 Prerequisites 1096
The prerequisites are in accordance with the relevant phases in which software configuration is 1097
applied. 1098
C.3.2 Further supporting information 1099
See applicable further supporting information of the relevant phases in which software configuration is 1100
applied. 1101
C.4 Requirements and recommendations 1102
C.4.1 The configuration data shall be specified to ensure the correct usage of the configurable 1103
software during the safety lifecycle. This shall include: 1104
a) the valid values of the configuration data; 1105
b) the intent and usage of the configuration data ; 1106
c) the range, scaling, units; and 1107
d) the interdependencies between different elements of the configuration data. 1108
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 48

ISO/DIS 26262-6:2016(E)
48 © ISO 2016 – All rights reserved

C.4.2 Verification of the configuration data shall be planned and performed in accordance with 1109
ISO 26262-8:2018, Clause 9 to ensure: 1110
a) the use of values within their specified range; and 1111
b) the compatibility with the other configuration data. 1112
NOTE The testing of the configured software is performed within the test phases of the software lifecycle (see 1113
Clause 9, Clause 10, Clause 11 and ISO 26262-4:2018, Clause 8). 1114
C.4.3 The ASIL of the configuration data shall equal the highest ASIL of the configurable software by 1115
which it is used. 1116
C.4.4 The verification of configurable software shall be planned, specified and executed in accordance 1117
with ISO 26262-8:2018, Clause 9. Configurable software shall be verified for the configuration data set 1118
that is to be used for the item development under consideration. 1119
NOTE Only that part of the embedded software whose behaviour depends on the configuration data is 1120
verified against the configuration data set. 1121
C.4.5 For configurable software a simplified software safety lifecycle in accordance with Figures C.2 1122
or C.3 may be applied. 1123
NOTE A combination of the following verification activities can achieve the complete verification of the 1124
configured software: 1125
a) “verification of the configurable software”, 1126
b) “verification of the configuration data”, and 1127
c) “verification of the configured software”. 1128
This is achieved by either 1129
 verifying a range of admissible configuration data in a) and showing compliance to this range in b), or 1130
 by showing compliance to the range of admissible configuration data in b) and performing c). 1131
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 49

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 49

Figure 3 — Variants of the reference phase model for software development 1132
with configurable software and different configuration data and different calibration data 1133
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 50

ISO/DIS 26262-6:2016(E)
50 © ISO 2016 – All rights reserved

Figure 4 — Variants of the reference phase model for software 1134
development with configurable software with the same configuration data and different 1135
calibration data 1136
C.4.6 The calibration data associated with software components shall be specified to ensure the 1137
correct operation and expected performance of th e configured software. This shall include: 1138
a) the valid values of the calibration data; 1139
b) the intent and usage of the calibration data; 1140
c) the range, scaling and units, if applicable, with their dependence on the operating state; 1141
d) the known interdependencies betwee n different calibration data; and 1142
NOTE Interdependencies can exist between calibration data within one calibration data set or between 1143
calibration data in different calibration data sets such as those applied to related functions implemented in 1144
the software of separate ECUs. 1145
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 51

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 51

e) the known interdependencies between configuration data and calibration data. 1146
NOTE Configuration data can have an impact on the configured software that uses the calibration data. 1147
C.4.7 The verification of the calibration data shall be p lanned, specified and executed in accordance 1148
with ISO 26262-8:2018, Clause 9. The verification of calibration data shall examine whether the 1149
calibration data is within its specified boundaries. 1150
NOTE Verification of calibration data can also be performed within application-specific software verification, 1151
or at runtime by the configurable software. 1152
C.4.8 The ASIL of the calibration data shall equal the highest ASIL of the software safety requirements 1153
it can violate. 1154
C.4.9 To detect unintended changes of safety-related calibration data, mechanisms for the detection 1155
of unintended changes of data as listed in Table C.1 shall be applied. 1156
Table 18 — Mechanisms for the detection of unintended changes of data 1157
Mechanisms
ASIL
A B C D
1a Plausibility checks on calibration data ++ ++ ++ ++
1b Redundant storage and comparison of calibration data + + + ++
1c Calibration data checks using error detecting codes a
+ + + ++
a Error detecting codes may also be implemented in the hardware in accordance with ISO 26262-5:2018.
C.4.10 The planning of the generation and application of calibration data shall specify: 1158
a) the procedures that shall be followed; 1159
b) the tools for generating calibration data; and 1160
c) the procedures for verifying calibration data. 1161
NOTE Verification of calibration data can include checking the value ranges of calibration data or the 1162
interdependencies between different calibration data. 1163
C.5 Work products 1164
C.5.1 Configuration data specification resulting from requirements C.4.1 and C.4.3. 1165
C.5.2 Calibration data specification resulting from requirement C.4.6. 1166
C.5.3 Configuration data resulting from requirement C.4.3. 1167
C.5.4 Calibration data resulting from requirement C.4.8. 1168
C.5.5 Software verification plan (refined) resulting from requirements C.4.2, C.4.4, C.4.7 and C.4.10. 1169
C.5.6 Verification specification (refined) resulting from requirements C.4.4 and C.4.7. 1170
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 52

ISO/DIS 26262-6:2016(E)
52 © ISO 2016 – All rights reserved

C.5.7 Verification report (refined) resulting from requirements C.4.1, C.4.4, C.4.7 and C.4.8. 1171
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 53

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 53

Annex D (informative) Freedom from interference between software elements 1172
D.1 Objectives 1173
The objective is to provide examples of faults that can cause interference between software eleme nts 1174
(e.g. software elements of different software partitions). Additionally, this Annex provides examples of 1175
possible mechanisms that can be considered for the prevention, or detection and mitigation of the listed 1176
faults. 1177
NOTE The capability and effectiven ess of the mechanisms used to prevent, or to detect and mitigate relevant 1178
faults is assessed during development. 1179
D.2 General 1180
D.2.1 Achievement of freedom from interference 1181
To develop or evaluate the achievement of freedom from interference between software elements, the 1182
effects of the exemplary faults, and the propagation of the possible resulting failures can be considered. 1183
D.2.2 Timing and execution 1184
With respect to timing constraints, the effects of faults such as those listed below can be considered for 1185
the software elements executed in each software partition: 1186
 blocking of execution, 1187
 deadlocks, 1188
 livelocks, 1189
 incorrect allocation of execution time , or 1190
 incorrect synchronization between software elements. 1191
EXAMPLE Mechanisms such as cyclic execution scheduling, fixed priority based scheduling, time triggered 1192
scheduling, monitoring of processor execution time, program sequence monitoring and arrival rate monitoring 1193
can be considered. 1194
D.2.3 Memory 1195
With respect to memory, the effects of faults such as those listed below can be considered for software 1196
elements executed in each software partition: 1197
 corruption of content, 1198
 inconsistent data (e.g. due to update during data fetch) , 1199
 stack overflow or underflow, or 1200
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 54

ISO/DIS 26262-6:2016(E)
54 © ISO 2016 – All rights reserved

 read or write access to memory allocated to another software element. 1201
EXAMPLE 1 Safety measures such as memory protection, parity bits, error -correcting code (ECC), cyclic 1202
redundancy check (CRC), redundant storage, restricted access to memory, static analysis of memory accessing 1203
software and static allocation can be used. 1204
EXAMPLE 2 Appropriate verification methods can be considered as detailed safety analysis to identify critical 1205
memories that protection mechanisms are used for. Results of static and semantic code analysis, control flow and 1206
data flow analysis can provide evidence for demonstrating freedom from interference. 1207
D.2.4 Exchange of information 1208
With respect to the exchange of information, the causes for faults or effects of faults such as those listed 1209
below can be considered for each sender or each receiver: 1210
 repetition of information, 1211
 loss of information, 1212
 delay of information, 1213
 insertion of information, 1214
 masquerade or incorrect addressing of information , 1215
 incorrect sequence of information, 1216
 corruption of information, 1217
 asymmetric information sent from a sender to multiple receivers , 1218
 information from a sender received by only a subset of the receivers , or 1219
 blocking access to a communication channel. 1220
NOTE The exchange of information between elements executed in different software partitions or different 1221
ECUs includes signals, data, messages, etc. 1222
EXAMPLE 1 Information can be exchanged using I/O-devices, data busses, etc. 1223
EXAMPLE 2 Mechanisms such as communication protocols, information repetition, loop back of information, 1224
acknowledgement of information, appropriate configuration of I/O pins, separated point-to-point unidirectional 1225
communication objects, unambiguous bidirectional communication objects, asynchronous data communication, 1226
synchronous data communicatio n, event-triggered data buses, event-triggered data buses with time-triggered 1227
access, time-triggered data busses, mini-slotting and bus arbitration by priority can be used. 1228
EXAMPLE 3 Communication protocols can contain information such as identifiers for communication objects, 1229
keep alive messages, alive counters, sequence numbers, error detection codes and error-correcting codes. 1230
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 55

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 55

EXAMPLE 4 Appropriate verification methods can be considered as detailed safety analysis to identify critical 1231
data exchange that protection mechanisms are used for. Results of static and semantic code analysis, control flow 1232
and data flow analysis can provide evidence for demonstrating freedom from interference. 1233
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 56

ISO/DIS 26262-6:2016(E)
56 © ISO 2016 – All rights reserved

Annex E (informative) Application of safety analyses and analyses of dependent 1234
failures at the software architectural level 1235
E.1 Objectives 1236
This Annex explains the possible application of safety analyses and d ependent failure analyses at the 1237
software architectural level. The examples provided in this Annex are intended to support the 1238
understanding of these analyses and provide guidance on their application . 1239
E.2 General 1240
E.2.1 Scope and purpose of the analyses 1241
The ability of the embedded software to provide the specified functions, behaviour and properties with 1242
the integrity as required by the assigned ASIL is also examined or confirmed by applying safety analyses 1243
or/and dependent failure analyses at the software architectural level. 1244
The suitability of embedded software to provide the specified functions and properties with the 1245
integrity required by the assigned ASIL is examined by: 1246
 identifying possible design weaknesses, conditions, faults or/and failures that could induce causal 1247
chains leading to the violation of safety requirements (e.g. using an inductive or deductive method); 1248
 analysing the consequences of possible faults, failures or/and causal chains on the functions and 1249
properties required for the software ar chitectural elements or/and the embedded software. 1250
The achieved level of independence or freedom from interference between the relevant software 1251
architectural elements is examined by analysing the software architectural design with respect to 1252
dependent fai lures, i.e.: 1253
 possible single events, faults or/and failures that may cause a malfunctioning behaviour of more 1254
than one of the software elements which require independence from each other (e.g. cascading 1255
and/or common cause failures and/or common mode failu res); 1256
 possible single events, faults or/and failures that may propagate from one software element to 1257
another inducing causal chains leading to the violation of safety requirements (e.g. cascading 1258
failures). 1259
Achievement of independence or freedom from interference between the software architectural 1260
elements can be required because of: 1261
 the application of an ASIL decomposition at the software level (see 6.4.3) , 1262
 the application of a design approach that combines architectural elements with different assign ed 1263
ASIL and/or elements with assigned ASIL and QM elements in a single software architectural 1264
design (see 7.4.6, 7.4.8 and 7.4.9), or 1265
 the implementation of a specific functional behaviour or specific properties . 1266
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 57

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 57

The role of safety analyses and analyses of dependent failures at software architectural level support s 1267
both design specification and design verification activities. 1268
Such analysis may also reveal incompleteness or inconsistencies with respect to given safety 1269
requirements. Therefore, consistency between existing safety requirements and safety measures , and 1270
newly identified requirements or measures , is beneficial. 1271
The results of safety analyses and analyses of dependent failures at software architectural level is also a 1272
basis for 1273
 the specification and implementation of effective safety mechanisms in the product , or 1274
 the determination of appropriate development-time safety measures, which are suitable either to 1275
prevent or to detect and control the relevant faults or failures identified during these analys es. 1276
Software safety
requirements
Requirements defining or refining
software properties
Requirements defining or refining
software functionalities
Technical safety requirements
and derived requirements related
to SW functionalities
Defined safety measures during SW development
OR
Requirements defining SW safety mechanisms
Attributes of
requirements
e.g. ASIL
Analyses provide evidence that the software architecture is suitable for required SW functionalities and properties with the respective ASIL
Technical safety requirements
and derived requirements related
to SW properties
Examples:
- Mechanisms to mitigate
effects of malfunctioning
behavior of SW components
- Specification of safetyrelated
monitoring functions
- Dedicated verification of
functional behavior of SW
components
Examples:
- Specification of safety-related
system functionality
implemented in SW
- Specification of diagnostic
functions to achieve
required DC for HW elements
Examples:
- Mechanisms to prevent or
detect and control
interference between
safety-related and nonsafety-related SW
components
- Mechnisms or measures to
ensure scheduling
properties
Examples:
- Required robustness of SW
against erroneous or missing
input signal from CAN
- Required tolerance of SW
against HW faults in memory
1277
1278
Figure E.1 — Relationship between software safety requirements and safety analyses 1279
To explain such a relationship Figure E.2 illustrates interference caused by a conflicting use of a shared 1280
resource (e.g. a shared processing element). In this example the QM software element interferes and 1281
prevents the timely execution of the ASIL software element . Examples are shown both without and 1282
with implementation of freedom from interference . 1283
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 58

ISO/DIS 26262-6:2016(E)
58 © ISO 2016 – All rights reserved

1284
QM software component
QM software component
ASIL software component
ASIL software component
execution time exceeds specified limits -
without timely detection
execution time exceed s specified limits -
with timely detection due to missed checkpoint
CP
CP
CP CP
CP
CPCP
Without freedom from interference measure :
With freedom from interference measure :
3 ms 3 ms
3 ms3 ms 3 ms
CP
3 ms
Legend:
Blocked software component
Executed software component
Interference
CP
CP
"Passed" program flow monitoring checkpoint
"Failed" program flow monitoring checkpoint
1285
Figure E.2 — Example of timing interference leading to a cascading failure 1286
Safety analyses and dependent failure analyses are intended to be applied at the software architecture 1287
level. Code level safety analyses (e.g. with respect to runtime errors such as divide by zero, access 1288
beyond array index boundary) are neither required nor are they considered appropriate for the 1289
following reasons: 1290
 The residual risk regarding these types of faults can be considered as sufficiently low provided that 1291
the software development approach as described in ISO 26262-6:2018 including the selection and 1292
application of suitable methods, approaches and development principles such as modelling and 1293
programming guidelines, design and implementation rules, model and/or code reviews, semantic 1294
or static analysis, unit and integration tests is applied carefully; 1295
 These types of faults in the implementation of the respective architectural elements usually induce 1296
the same malfunctioning behaviour of the element at its interfaces as analysed at architectural 1297
level. Therefore the analyses at the higher level already provide the evidence that the selected 1298
technical safety mechanisms are effective or that the development -time safety measures are 1299
sufficient to ensure an adequate argument. 1300
Exceptional cases may exist in which a code level analysis may be appropriate because of a specific 1301
situation, a specific software safety concept or a specific software argument, but a decision to perform 1302
such analyses usually are related to weaknesses in architectural design concepts. 1303
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 59

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 59

E.2.2 Relationship between safety analyses and dependent failure analyses at the 1304
system, hardware and software level 1305
Safety analyses and dependent failure analyses are applied during the different phases of the 1306
development lifecycle with possible dependencies and interactions. 1307
In a separation of concern approach initiated at the system level it is usually the responsibility of 1308
hardware development to ensure that the processing unit possesses a sufficient low residual risk 1309
regarding hardware faults. Therefore the software safety analysis and software dependent failure 1310
analysis at architectural level can be performed without considering ha rdware faults. 1311
However under some conditions (e.g. insufficient diagnostic coverage regarding transient faults or 1312
random hardware faults of the processing element) it may be appropriate to consider the negative 1313
influence of hardware faults. In such cases specific analyses regarding the specific hardware /software 1314
interaction are performed considering for example: 1315
 The specific software mapping to the different processing units of multi -core systems; 1316
 The detailed design of the processing element on which the software is executed regarding the 1317
static or dynamic aspects of the software architectural design ; 1318
 The achievable diagnostic coverage of the selected software safety mechanisms regarding software-1319
relevant hardware faults . 1320
NOTE Analysis of a specific hardware/software interaction can be supported by fault injection techniques 1321
(see also ISO 26262-11:2018 Clause 4.8). 1322
E.2.3 Safety analyses in a distributed software development (including SEooC 1323
development of software elements) 1324
Embedded software and its software elements are often developed as a distributed development, by 1325
more than one organization (including OEM, Tier 1, Tier 2, silicon vendor or software vendor) and even 1326
out of context (e.g. basic software, hardware-related software, COTS software or operating systems 1327
developed as SEooC). 1328
Important aspects for meaningful analyses in distributed software developments are: 1329
 Definition and agreement about the scope of the analyses and the procedures or methods used for 1330
performing the analyses either separately or jointly (including exchange of 1331
information/documentation or confirmation of assumptions); 1332
 Definition and agreement about the interfaces when using a modular approach for the analyses (e.g. 1333
agreed software fault models at the interfaces between the different scopes, the approach used or 1334
the exchanged information/documents); 1335
 Definition and agreement about the verification of the analyses (e.g. joint reviews). 1336
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 60

ISO/DIS 26262-6:2016(E)
60 © ISO 2016 – All rights reserved

E.3 Topics for performing analyses 1337
The software architecture design is an important mean s to manage the complexity of the software 1338
development by defining and providing the design aspects (e.g. using more formal ized or less 1339
formalized notations) needed to develop and integrate the software. From the safety perspective, it is 1340
important to summarize the appropriate topics and levels of detail in such way that the achievement of 1341
functional safety can be argued. 1342
Common topics to achieve meaningful analyses are the determination of: 1343
 the specific goals for the analyses (e.g. derive them from required functionality/properties of 1344
software such as robustness or deterministic execution) – see E.3.1; 1345
 the scope and boundaries regarding the specific architectural design to be analysed (incl uding 1346
interfaces) – see E.3.2; 1347
 the method applied to conduct the analyses – see E.3.3; 1348
 means or criteria to support the significance, objectivity and rigo ur of the analyses – see E.3.4. 1349
Further guidance on these topics and possible dependencies is provided in the following paragraphs. 1350
E.3.1. Determination of the specific goals for the analyses 1351
Besides the general goals provided by Clause 7 and ISO 26262-9, specific goals for the analyses can be 1352
derived from the safety requirements assigned to the software or to software components. 1353
EXAMPLE Application-specific unwanted events or malfunctioning behaviour caused by the violation of the 1354
safety-related functionalities or properties assigned to the software within the scope of the analyses. 1355
E.3.2 Determination of the scope and boundaries regarding the s pecific architectural 1356
design to be analysed 1357
Depending on the goals of the analyses the scope is determined setting the focus to the relevant parts of 1358
the architectural design. 1359
The scope of the analyses can be influenced by : 1360
 the relevant scope of delivery and the respective responsibilities as defined in the DIA , 1361
 the specific goals of the analyses, 1362
 architectural strategies supported by “good” design principles (see 7.4. 3), or by 1363
 properties required from the architectural desi gn resulting from higher level safety concepts in 1364
respect of the achievement of freedom from interference or sufficient independence . 1365
EXAMPLE 1 Appropriate end-to-end data protection mechanisms can be used as an argument that the basi c 1366
software can treated as a “black box” when considering the exchange of safety-related data with external senders 1367
or receivers during a safety analyses. 1368
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 61

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 61

EXAMPLE 2 Dependencies of software functions with operating modes such as a safety-related function which 1369
is only “active” in a mode w here the other “critical” function is safely deactivated by a safe mode switching 1370
function. 1371
SWC Y
Monitoring function N Comparator
function O
Function O
Guard M
Software Component (SWC) with
allocated safety requirements
SWC V
Signal A1 An
SWC U
SWC X
Software Component (SWC) which
is in the scope of the analyses
Software Component (SWC) which is excluded from the
scope of the analyses based on an argument
Functional interaction (e.g. data flow)
Legend:
1372
Figure E.3 — Scope of the analyses with respect to the safety argumentation 1373
Figure E.3 shows an example where according to the safety concept of SWC X the safety -related 1374
functionalities are implemented only in SWC U and controlled by a monitoring function implemented in 1375
SWC V. Determination of the scope of the analyses on the architecture c larifies the required safety 1376
arguments and supports the further steps shown in the following sections. In this example, SWC Y can 1377
be exempted from a detailed analysis unless the analyses of SWC V and SWC U is unable to confirm the 1378
assumed freedom -from-interference (e.g. because of weaknesses in the Guard M regarding cascading 1379
failures with respect to erroneous signals A 1 to An). 1380
E3.3. Determination of the method applied to conduct the analyses 1381
Due to the specific nature of software (e.g. no random faults due to wear out or ag eing and lack of a 1382
mature probabilistic method) methods established for such analyses at the system or hardware level 1383
often cannot be transferred to software without modifications or will not deliver results that are as 1384
meaningful . 1385
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 62

ISO/DIS 26262-6:2016(E)
62 © ISO 2016 – All rights reserved

Common factors of safety analyses methods are that they usually consist of a means to enforce a 1386
structured approach, to store the gained knowledge and to draw conclusions. 1387
EXAMPLE Syntax and semantics to describe faults and their dependencies in a fault tree or the function net 1388
or list of elements or Risk Priority Number used in an FMEA. 1389
The basis for the safety analyses and dependent failure analyses is the architectural design describing 1390
the static aspects (e.g. expressed by a block diagram showing the funct ional elements and their 1391
interfaces/relationships) as well as the dynamic aspects (e.g. expressed by sequence, timing or 1392
scheduling diagrams or state charts) of the related software. 1393
The safety analyses and dependent failure analyses according to the arch itectural design can follow 1394
functional and/or processing chains considering both static and dynamic aspects. During such analyses 1395
software fault models or questions such as “which specific fault or failure originating from or impacting 1396
this software compon ent can lead to the violation of an assigned safety requirement” can be used to 1397
identify safety-related weaknesses in the design. 1398
Refinements of architectural design can support the safety analyses and dependent failure analyses for 1399
the adequate level of details in order to identify lower level s of faults or failures. 1400
The level of detail required during such an analyses is related to: 1401
 adequate selection of the scope during refinements of the architecture design, 1402
 properties that support to achieve the specific goals of the analyses, 1403
 the argument based on the architectural strategy supported by “good” design principles (see 7.4. 3), 1404
or 1405
 the argument resulting from achieved freedom from interference or sufficient independence . 1406
E.3.4 Aspects to support the meaningfulness, objectivity and rigo ur of the analyses 1407
E.3.4.1 Dependent failure initiators for DFA in accordance with ISO 26262-9:2018 1408
In Annex C of ISO 26262–9:2018 dependent failure initiators are defined that can be used to improve 1409
the results of a dependent failure analysis performed on functional or processing chains of the software 1410
that are required to be independent. In practice, at the software level the following amplifications for 1411
the dependent failure initiators can be used for guiding the analyses (e.g. a checklist). 1412
Table E.1 – Analysis at the software level 1413
Type of independence violation Examples for the software level
Shared resource Same hardware element instance
used by the two functions which
are therefore affected by the
failure or unavailability of that
shared resource, i.e. from a
• Software component used by more than
one other software component, e.g.
– reused standard software modules
– libraries
– middleware
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 63

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 63

physical perspective. – basic software
– operating system including
scheduler
– communication stack
– parameter file
– execution time
– allocated memory
Components of
identical type
Identical components may fail
with higher probability
simultaneously (common mode
failures)

• Reused standard software modules
• Standard library components
• Middleware
• Operating system
• Communication stack
• Parameter set
Communication
between the two
functions
Receiving function is affected by
information that is false, lost,
sent multiple times, or in the
wrong order etc. from the sender
• Information passed via argument through
a function call, or via writing/reading a
variable being global to the two software
functions (data flow)
• Data or message corruption/ repetition*/
loss* / delay*/ masquerading or incorrect
addressing of information*
• Insertion*/ sequence of information*
• Corruption of information*
• Asymmetric information sent from a
sender to multiple receivers*
• Information from a sender received by
only a subset of the receivers*
• Blocking access to a communication
channel*
Shared information
inputs
Same information consumed by
the two functions even in
absence of shared resources, i.e.
from a functional perspective
• Calibration parameters
• Constants, or variables, being global to the
two software functions
• Basic software passes data (read from
hardware register and converted into
logical information) to two application
software functions
• Data / function parameter arguments /
messages delivered by software function t o
more than one other function
See also “Communication between the two
functions”
Environmental
influences
Common external environmental
disturbance affects both
functions

n.a.
Unintended impact Two functions affecting each
other’s elements directly via an
implicit, that is unintended,
• Memory / corruption of content*
– “wild pointers“ (including
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 64

ISO/DIS 26262-6:2016(E)
64 © ISO 2016 – All rights reserved

interface. program counter stack)
– memory mis-allocation
– memory leaks
– read/write access to memory
allocated to another software
element*
– buffer under-/overflow
• Timing and execution
– deadlocks*
– livelocks*
– incorrect allocation of execution
time*
– incorrect synchronization
between software elements*
Systematic coupling Systematic cause s from human
or tool errors can lead to the
simultaneous failure of more
than one function

• Manufacturing fault / repair fault (e.g. false
flashing, false calibration reference for
sensors)
• Non-diverse development approaches
including:
– same software tools (e.g. IDE,
compiler, linker)
– same algorithms
– same programming and/or
modelling language used
– same complier/linker used
• Same personnel
• Same social -cultural context (even if
different personnel)
• Development fault, e.g.
– human error
– insufficiently qualified personnel
– process weaknesses
– insufficient methods
* These examples are taken from Annex D on freedom from interference between software elements. In that
respect, the dependent failure initiators Unintended Impact and Communication represent causes of violation of
freedom from interference for software.
1414
E.3.4.2 Use content of Annex D 1415
Annex D provides software fault models to be considered when analysing interference between 1416
software elements, for example with respect to: 1417
 timing and execution , 1418
 memory, or 1419
 exchange of information. 1420
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 65

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 65

E.3.2 Analyses supported by using guide words 1421
Guide words can be used to systematically examine the possible deviations from a specific design intent 1422
and to determine their possible consequences. Guide words are used to generate qu estions for the 1423
examination of a specific functionality or property of the architectural element during the analyses. 1424
When using guide words the analyses of the specific functions or properties for each design element are 1425
repeated with each guide word unti l the predetermined guide word s have all been examined. Thus 1426
guide words are a mean to conduct such analyses systematically and to provide an argument for 1427
completeness. 1428
Data flow or similar diagrams are often used to describe the functional aspects of the software 1429
architectural design (e.g. functional or processing chains). Figure E.4 shows interaction of the software 1430
components Y, U and V and the related functional or processing chain (e.g. based on data flow) . 1431
Guide words can be used to identify weakness es, faults, and failures. The selection of suitable guide 1432
words depends on the characteristics of the examined functions, behaviour, properties, interfaces or 1433
exchanged data. 1434
SWC Y
Monitoring function N Comparator
function O
Function O
Guard M
Software Component (SWC) with
allocated safety requirements
SWC V
Signal A1 An
SWC U
SWC X
Software Component (SWC) which
is in the scope of the analyses
Software Component (SWC) which is excluded from the
scope of the analyses based on an argument
Functional interaction (e.g. data flow)
Legend:
1435
Figure E.4 — Block diagram of software architectural design 1436
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 66

ISO/DIS 26262-6:2016(E)
66 © ISO 2016 – All rights reserved

Using the design intent as a bas is the guide words are derived by combining design attributes, the 1437
related guide words and their interpretation in order to achieve a common understanding for the 1438
analyses. Table E.2 shows examples for the selection of guide words. 1439
Table E.2 – Example for the selection of guide words 1440
Examined
functionality
or property
Guide word Interpretation Additional references
signal flow
(A1….An)
After / Late Signal too late or out of
sequence.
D.2.4 Delay of information
D.2.4 Incorrect sequence of information
D.2.4 Blocking access to communication channel
D.2.2 Blocking of execution
D.2.2 Deadlocks
D.2.2 Livelocks
D.2.2 Incorrect allocation of execution time
D.2.2 Incorrect synchronization between
software components
Message was sent or received or queued too late
The initial message was lost and the
communication subsystem attempted a re-try
without notifying the application
Sending or receiving process was blocked too
long
Before /
Early
Signal too early or out of
sequence.
D.2.4 Incorrect sequence of information
D.2.2 Incorrect synchronization between
software components
Message sent too early
Sending process scheduled too often
No No signal D.2.2 Blocking of execution
D.2.2 Deadlocks
D.2.2 Livelocks
D.2.2 Incorrect synchronization between
software components
Signal value More Signal value exceeds permitted
range

Less Signal value falls bellows the
permitted range

1441
During the analysis guide words can be applied in a deductive or inductive approach. During the 1442
analyses a further refinement of the guide words is possible if required (e.g. to better match specific 1443
design aspects). 1444
During the analyses the guide words are used according to the selected method (e.g. following function 1445
or processing chains) to generat e the questions that will examine the design and reveal possible 1446
weaknesses and their consequences. An e xample for a guide word supported analys is of the 1447
Signals A 1 … An is shown in Table E.3. 1448
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 67

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 67

Table E.3 – Example for a guide word supported analysis 1449
Guide
word
Interpretation Consequence Safety measure Required activity
More Signal value exceeds
permitted range
Function O will
generate erroneous
output
Guard M limits Signal
A1…An to the permitted
maximum range
Implement Guard M
Less Signal value falls
bellows the permitted
range
Function O will
generate erroneous
output
Guard M limits Signal
A1…An to the permitted
maximum range
Implement Guard M
Other
than
Values of signals A1 …
An are inconsistent
Function O will
generate erroneous
output
Guard M checks A1…An
for consistency using
physical dependencies
Implement Guard M
E.4 Mitigation strategy for weaknesses identified during safety and dependent 1450
failure analyses 1451
The result of safety analyses and dependent failure analyses at the software architectural level allow the 1452
selection of adequate safety measures to prevent or to detect and control faults and failures with the 1453
potential to violate safety requirements or safety goals. Figure E.5 shows a possible strategy for the 1454
determination of suitable safety measures based on such results. 1455
1456
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 68

ISO/DIS 26262-6:2016(E)
68 © ISO 2016 – All rights reserved

Start
Determine software architectural
design and appropriate software
fault model (s) for analyses
Step-by-step analysis of each
architectural element against
each fault of fault model (s)
Have all faults
for the selected architectural
element been analysed ?
Have all
architectural elements been
analysed?
Is a change of the
software architectural design
required?
End
Update the software
architectural design yes
no
yes
yes
Select next fault of fault model (s)
Select next element of
architectural design
no
no
Evaluate the criticality of the
fault, based on whether it can
violate a safety goal or safety
requirement
Are
development -time safety
measures sufficient to mitigate the
fault, given its criticality ?
Determine effective
safety mechanism (s)
yes
no
Provide rationale for sufficiency
(including introduction of
additional measures )
1457
Figure E.5 — Decision graph for “acceptance” of counter measures (i.e. technical on-line safety 1458
mechanisms vs. development-time safety measures) 1459
As shown in Figure E.5 it is not always necessary to implement safety mechanisms which are able to 1460
prevent or detect and control such faults or failures at runtime. 1461
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 69

ISO/DIS 26262-6:2016(E)
© ISO 2016 – All rights reserved 69

Choosing a suitable mitigation strategy can be based on aspects such as: 1462
 complexity of the software architectura l design (e.g. number of interfaces and software 1463
components) and/or the software components (e.g. number of assigned requirements or size of the 1464
component), or 1465
EXAMPLE For more complex software a n argument based solely on development-time measures is less 1466
suitable. 1467
 effectiveness of development -time safety measures in order to prevent the relevant faults or 1468
failures (e.g. effectiveness of verification activities considering the software fault model s used 1469
during the analyses) . 1470
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.

## Page 70

ISO/DIS 26262-6:2016(E)
70 © ISO 2016 – All rights reserved

Bibliography 1471
[1] ISO/IEC 12207, Systems and software engineering — Software life cycle processes 1472
[2] IEC 61508 (all parts), Functional safety of electrical/electronic/programmable electronic safety -1473
related systems 1474
[3] MISRA C:2012, Guidelines for the use of the C language in critical systems, ISBN 978-1-906400-10-1, 1475
MIRA, March 2013 1476
[4] MISRA AC AGC, Guidelines for the application of MISRA -C:2004 in the context of automatic code 1477
generation, ISBN 978 -1-906400-02-6, MIRA, November 2007 1478
[5] Bieman, J.M.; Dreilinger, D.; Lijun Lin, “Using fault injection to increase software test coverage, ” in 1479
Software Reliability Engineering, 1996. Proceedings., Seventh International Symposium on , vol., 1480
no., pp.166-174, 30 Oct-2 Nov 1996 doi: 10.1109/ISSRE.1996.558776 1481
[6] Jia, Y., Merayo, M., and Harman, M. (2015) Introduction to the special issue on Mutation Testing. 1482
Softw. Test. Verif. Reliab., 25: 461 –463. 1483
[7] ISO 26262-12:2018, Road Vehicles — Functional safety— Part 12: A daptation of ISO 26262 for 1484
Motorcycles 1485
Licensed to MDS Technology CO.,LTD. / Seungyong Han (seungyong@mdstec.com)
ISO Store Order: OP-186844 / Downloaded: 2016-12-26
Single user licence only, copying and networking prohibited.
