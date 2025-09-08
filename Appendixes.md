# Appendix for the paper

## Note
This appendix is provided in text form for double-blind review. Appendixes will be available in Arxiv via link after the acceptance of the paper.

## Class lists

### L
```
void
wall
floor
chair
door
table
picture
cabinet
cushion
window
sofa
bed
curtain
chest_of_drawers
plant
sink
stairs
ceiling
toilet
stool
towel
mirror
tv_monitor
shower
column
bathtub
counter
fireplace
lighting
beam
railing
shelving
blinds
gym_equipment
seating
board_panel
furniture
appliances
clothes
objects
misc
```

### L_q
```
wall
floor
chair
door
table
picture
cabinet
cushion
window
sofa
bed
curtain
chest_of_drawers
plant
sink
stairs
ceiling
toilet
stool
towel
mirror
tv_monitor
shower
column
bathtub
counter
fireplace
lighting
beam
railing
shelving
blinds
gym_equipment
board_panel
clothes
```

## Prompts

### Synonym prompt

```
give a list of <num> strings,
that are individual nouns,
containing a combination of the following:
synonyms for X,
very similar things than X,
things that look similar than X, even if they are not X,
if X is a subclass, superclass for X,
possible interpretations of X,
possible related items of X,
if verb or activity or a wish, items required to do X,
items or activities that are usually associated with X,
and abstracted and generalized versions of the all above.
Focus on the phrasing. If asked for example 'A on B',
return A and only words related to A.
All should be likely found in an average real environment, depicting: Y.
Return the results as a comma-separated list, do not number the
results or use any other formatting.

X is a real existing thing, interpret it such. If it is too specific,
like 'REDACTED's coffee mug',
you can abstract it to 'coffee mug' etc. 'Coffee mug' is a subclass of
'mug', so also 'mug' is ok to be included.
Y = <env>.
X = <query>.
```

### Synonym system prompt

```
You are an expert image segmentation assistant.
Your job is to figure out words and phrases that help segmenting objects from images.
All of the images are existing, and nothing is made up.
Try to keep the solutions as simple as possible. Use common words.
Use simple words first, complex later. Do not use too specific terminology.
For example: Instead of 'corvus corax' first say 'bird' then 'raven'
When recommending synonyms use parts of the word. For example: 'bird' has 'wings', 'beak' and 'feathers'.
Focus on the phrasing. If asked for example 'A on B', 'A in B', 'A behind B', return A and only words related to A
Ignore typos in queries.
You've got this, we believe in you!
```

### Antonym prompt

```
give a list of <num> strings, that can be individual words or phrases,
containing a combination of the following:
antonyms for X,
descriptions of background regions in images,
things that use X, but are not X,
if X is an item, the user of X,
things that could be confused to X,
or are usually nearby X but not X itself,
or closely related to X but are not X,
or are helpful for distinguishing X from nearby items,
or are other normal things than X found in a image depicting: Y,
and abstracted and generalized versions of the all above.
Focus on the phrasing. If asked for example 'A on B', return B and only words related to B.
All should be likely found in an average real environment, depicting: Y.
Return the results as a comma-separated list, do not number the results or use any other formatting.

Do not use abstract terms such as 'not X' or if X is specific: 'other X'.
Do not name parts of X.
Additionally, do not use anything on this list: <synonyms>
Y = <env>.
X = <query>.
```

### Antonym system prompt

```
You are an expert image segmentation assistant.
Your job is to figure out words and phrases that help segmenting the background from images.
All of the images are existing, and nothing is made up.
Try to keep the solutions as simple as possible. Use common words.
Use simple words first, complex later. Do not use too specific terminology.
For example: Instead of 'corvus corax' first say 'bird' then 'raven'.
Focus on the phrasing. If asked for example 'A on B', 'A in B', 'A behind B', return B and only words related to B.
NEVER return parts of the queried item. For example: 'bird' has 'wings', 'beak' and 'feathers'. NEVER return these.
NEVER return the base class for specific query: i.e. 'bird' if asked for 'raven'. NEVER do this.
NEVER return 'not' + queries word.
NEVER return 'other ...'
NEVER return plurals
Ignore typos in queries.
You've got this, we believe in you!
```