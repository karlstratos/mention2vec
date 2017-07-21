Mention2Vec
============

__Mention2Vec__ is a neural architecture for named-entity recognition (NER). It frames NER as multitasking boundary detection and entity classification.

Highlights
-----------
1. Mention2Vec is a good NER model. In particular, it can achieve 90.9 test F1 score on CoNLL 2003. To replicate the result, train the model by typing
```
python mention2vec.py /tmp/m2v ../data/conll2003.train --train --dynet-mem 2048 --dynet-seed 42 --dev ../data/conll2003.dev --emb sskip.100.vectors
```
where `sskip.100.vectors` is the pre-trained word embeddings used in [Lample et al. (2016)](https://arxiv.org/pdf/1603.01360.pdf). Then run the model on the test data by typing
```
python mention2vec.py /tmp/m2v ../data/conll2003.test --pred /tmp/pred
```
The path to prediction file is optional.

2. Mention2Vec can crank out context-sensitive representations of predicted entities. To obtain them, simply add `--entemb` flag when you run the model at test time:
```
python mention2vec.py /tmp/m2v ../data/conll2003.test --pred /tmp/pred --entemb /tmp/entemb
```
The lines in /tmp/entemb look like [sentence with entity mark] and [embedding values]. You can then use the provided script to see nearest neighbors.

### Some nearest neighbor examples

`python display_nearest_neighbors.py /scratch/m2v/entemb.dev`

`Read 5802 embeddings of dimension 4`

`Type a word (or just quit the program):`

`Lauck__,__from__{{Lincoln}}__,__Nebraska__,__yelled__a__tirade__of__abuse__at__the__court__after__his__conviction__for__inciting__racial__hatred__.`

		1.0000		The__boat__had__been__missing__since__Monday__afternoon__when__it__left__the__tiny__island__of__Gorgona__off__Colombia__'s__southwest__coast__with__sightseers__for__a__return__trip__to__{{Narino}}__province__,__near__the__border__with__Ecuador__.
		1.0000		ST__LOUIS__AT__{{COLORADO}}
		1.0000		Gephardt__,__55__,__the__son__of__a__milkman__from__a__working__class__district__of__{{St.__Louis}}__,__is__a__consummate__congressional__insider__,__sufficiently__skilled__in__compromise__and__the__ways__of__the__legislature__to__manage__the__often-unruly__House__Democrats__.
		1.0000		She__said__the__train__was__travelling__at__54__mph__when__it__crashed__into__the__truck__,__which__was__crossing__the__tracks__onto__a__dirt__road__in__the__rural__area__bordering__the__{{Northfield__Mountains}}__.
		1.0000		The__sixth-ranked__Ivanisevic__,__who__lost__in__the__final__at__{{Indianapolis}}__to__world__number__one__Pete__Sampras__of__the__U.S.__last__Sunday__,__made__a__quick__getaway__after__his__loss__but__did__say__:__"__Something__was__not__there__when__I__arrived__-LPR-__in__Toronto__-RPR-__.

`Type a word (or just quit the program):`

`In__another__letter__dated__January__1865__,__a__well-to-do__Washington__matron__wrote__to__{{Lincoln}}__to__plead__for__her__son__,__who__faced__a__dishonourable__discharge__from__the__Army__.__"`

		1.0000		Norris__said__Tuesday__'s__surgery__involved__placing__five__balloons__in__{{DeJesus}}__'s__forehead__,__shoulders__and__the__back__of__her__neck__and__partially__filling__them__with__a__saline__solution__.
		1.0000		Specter__met__Crown__Prince__Abdullah__and__Minister__of__Defence__and__Aviation__Prince__{{Sultan}}__in__Jeddah__,__Saudi__state__television__and__the__official__Saudi__Press__Agency__reported__.
		1.0000		{{Green}}__and__black__face__paint__completed__his__disguise__.
		1.0000		A__few__kilometres__-LPR-__miles__-RPR-__down__the__two-lane__road__which__passes__{{Odnosum}}__'s__farm__,__where__the__occasional__horse-drawn__buggy__passes__by__,__is__the__1,700-hectare__-LPR-__4,200-acre__-RPR-__Shevchenko__collective__farm__,__built__next__to__a__village__still__neat__and__tidy__despite__post-Soviet__decay__.
		1.0000		Tapie__,__the__target__of__a__blizzard__of__legal__actions__over__his__now-destroyed__business__empire__and__the__Marseille__soccer__team__he__once__ran__,__has__a__starring__role__in__{{Lelouche}}__'s__"__Homme__,__femmes__:__mode__d'emploi__"__-LPR-__Men__,__women__:__instructions__for__use__-RPR-__.

`Type a word (or just quit the program):`

`Baltimore__has__won__16__of__its__last__22__games__to__pull__within__five__games__of__the__slumping__{{New__York__Yankees}}__in__the__American__League__East__Division__.`

		1.0000		Note__-__LG__and__OB__,__Ssangbangwool__and__{{Hanwha}}__played__two__games__.
		1.0000		Grobbelaar__now__plays__for__English__second__division__leaders__Plymouth__Argyle__after__years__in__the__top__flight__with__{{Liverpool}}__and__Southampton__.
		1.0000		At__Colchester__:__Essex__beat__{{Gloucestershire}}__by__an__innings__and
		1.0000		When__Arafat__returned__to__the__podium__,__he__said__senior__{{PLO}}__negotiator__Mahmoud__Abbas__,__better__known__as__Abu__Mazen__,__and__Netanyahu__aide__Dore__Gold__could__meet__on__Thursday__.
		1.0000		2__-__Miladin__Becanovic__-LPR-__Lille__-RPR-__,__Enzo__Scifo__-LPR-__{{Monaco}}__-RPR-__,


Reference
---------
[Entity Identification as Multitasking](?)