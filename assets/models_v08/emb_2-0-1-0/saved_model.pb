??+
?$?#
.
Abs
x"T
y"T"
Ttype:

2	
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( " 
Ttype:
2	"
Tidxtype0:
2	
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
,
Exp
x"T
y"T"
Ttype:

2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
9
	IdentityN

input2T
output2T"
T
list(type)(0
,
Log
x"T
y"T"
Ttype:

2
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
8
Pow
x"T
y"T
z"T"
Ttype:
2
	
?
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
@
ReadVariableOp
resource
value"dtype"
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
@
Softplus
features"T
activations"T"
Ttype:
2
?
SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	?
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.10.12v2.10.0-76-gfdfc646704c8??(
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
p
Const_1Const*&
_output_shapes
:*
dtype0*)
value B"  ??    
h
Const_2Const*
_output_shapes

:*
dtype0*)
value B"              
I
Const_3Const*
_output_shapes
: *
dtype0*
value	B :
I
Const_4Const*
_output_shapes
: *
dtype0*
value	B :
X
Const_5Const*
_output_shapes
:*
dtype0*
valueB"      
?
2Adam/embedding_normal_diag_1/untransformed_scale/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*C
shared_name42Adam/embedding_normal_diag_1/untransformed_scale/v
?
FAdam/embedding_normal_diag_1/untransformed_scale/v/Read/ReadVariableOpReadVariableOp2Adam/embedding_normal_diag_1/untransformed_scale/v*
_output_shapes

:*
dtype0
?
0Adam/embedding_normal_diag/untransformed_scale/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*A
shared_name20Adam/embedding_normal_diag/untransformed_scale/v
?
DAdam/embedding_normal_diag/untransformed_scale/v/Read/ReadVariableOpReadVariableOp0Adam/embedding_normal_diag/untransformed_scale/v*
_output_shapes

:*
dtype0
?
 Adam/embedding_normal_diag/loc/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*1
shared_name" Adam/embedding_normal_diag/loc/v
?
4Adam/embedding_normal_diag/loc/v/Read/ReadVariableOpReadVariableOp Adam/embedding_normal_diag/loc/v*
_output_shapes

:*
dtype0
?
2Adam/embedding_normal_diag_1/untransformed_scale/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*C
shared_name42Adam/embedding_normal_diag_1/untransformed_scale/m
?
FAdam/embedding_normal_diag_1/untransformed_scale/m/Read/ReadVariableOpReadVariableOp2Adam/embedding_normal_diag_1/untransformed_scale/m*
_output_shapes

:*
dtype0
?
0Adam/embedding_normal_diag/untransformed_scale/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*A
shared_name20Adam/embedding_normal_diag/untransformed_scale/m
?
DAdam/embedding_normal_diag/untransformed_scale/m/Read/ReadVariableOpReadVariableOp0Adam/embedding_normal_diag/untransformed_scale/m*
_output_shapes

:*
dtype0
?
 Adam/embedding_normal_diag/loc/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*1
shared_name" Adam/embedding_normal_diag/loc/m
?
4Adam/embedding_normal_diag/loc/m/Read/ReadVariableOpReadVariableOp Adam/embedding_normal_diag/loc/m*
_output_shapes

:*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
?
exponential_similarity/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameexponential_similarity/beta
?
/exponential_similarity/beta/Read/ReadVariableOpReadVariableOpexponential_similarity/beta*
_output_shapes
: *
dtype0
?
exponential_similarity/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameexponential_similarity/gamma
?
0exponential_similarity/gamma/Read/ReadVariableOpReadVariableOpexponential_similarity/gamma*
_output_shapes
: *
dtype0
?
exponential_similarity/tauVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameexponential_similarity/tau
?
.exponential_similarity/tau/Read/ReadVariableOpReadVariableOpexponential_similarity/tau*
_output_shapes
: *
dtype0
?
Dstochastic_behavior_model/rank_similarity/distance_based/minkowski/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDstochastic_behavior_model/rank_similarity/distance_based/minkowski/w
?
Xstochastic_behavior_model/rank_similarity/distance_based/minkowski/w/Read/ReadVariableOpReadVariableOpDstochastic_behavior_model/rank_similarity/distance_based/minkowski/w*
_output_shapes
:*
dtype0
n
minkowski/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameminkowski/rho
g
!minkowski/rho/Read/ReadVariableOpReadVariableOpminkowski/rho*
_output_shapes
: *
dtype0
?
embedding_normal_diag_1/locVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*,
shared_nameembedding_normal_diag_1/loc
?
/embedding_normal_diag_1/loc/Read/ReadVariableOpReadVariableOpembedding_normal_diag_1/loc*
_output_shapes

:*
dtype0
f
	kl_annealVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	kl_anneal
_
kl_anneal/Read/ReadVariableOpReadVariableOp	kl_anneal*
_output_shapes
: *
dtype0
?
+embedding_normal_diag_1/untransformed_scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*<
shared_name-+embedding_normal_diag_1/untransformed_scale
?
?embedding_normal_diag_1/untransformed_scale/Read/ReadVariableOpReadVariableOp+embedding_normal_diag_1/untransformed_scale*
_output_shapes

:*
dtype0
?
)embedding_normal_diag/untransformed_scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*:
shared_name+)embedding_normal_diag/untransformed_scale
?
=embedding_normal_diag/untransformed_scale/Read/ReadVariableOpReadVariableOp)embedding_normal_diag/untransformed_scale*
_output_shapes

:*
dtype0
?
embedding_normal_diag/locVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_nameembedding_normal_diag/loc
?
-embedding_normal_diag/loc/Read/ReadVariableOpReadVariableOpembedding_normal_diag/loc*
_output_shapes

:*
dtype0
?
#serving_default_2rank1_stimulus_setPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
serving_default_agent_idPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCall#serving_default_2rank1_stimulus_setserving_default_agent_idConst_5Const_4embedding_normal_diag/loc)embedding_normal_diag/untransformed_scaleembedding_normal_diag_1/loc+embedding_normal_diag_1/untransformed_scale	kl_annealConst_3minkowski/rhoDstochastic_behavior_model/rank_similarity/distance_based/minkowski/wexponential_similarity/betaexponential_similarity/tauexponential_similarity/gammaConst_2Const_1Const*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*,
_read_only_resource_inputs


*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_118476

NoOpNoOp
?X
Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*?W
value?WB?W B?W
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
behavior
		optimizer


signatures*
J
0
1
2
3
4
5
6
7
8
9*

0
1
2*
* 
?
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
trace_0
trace_1
trace_2
trace_3* 
6
trace_0
trace_1
 trace_2
!trace_3* 
_
"	capture_0
#	capture_1
$	capture_7
%
capture_13
&
capture_14
'
capture_15* 
?
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses
.percept

/kernel
0percept_adapter
1kernel_adapter
2
_z_q_shape
3
_z_r_shape*
?
4iter

5beta_1

6beta_2
	7decay
8learning_ratem?m?m?v?v?v?*

9serving_default* 
YS
VARIABLE_VALUEembedding_normal_diag/loc&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE)embedding_normal_diag/untransformed_scale&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE+embedding_normal_diag_1/untransformed_scale&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUE	kl_anneal&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEembedding_normal_diag_1/loc&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEminkowski/rho&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEDstochastic_behavior_model/rank_similarity/distance_based/minkowski/w&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEexponential_similarity/tau&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEexponential_similarity/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEexponential_similarity/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
5
0
1
2
3
4
5
6*

0*

:0
;1*
* 
* 
_
"	capture_0
#	capture_1
$	capture_7
%
capture_13
&
capture_14
'
capture_15* 
_
"	capture_0
#	capture_1
$	capture_7
%
capture_13
&
capture_14
'
capture_15* 
_
"	capture_0
#	capture_1
$	capture_7
%
capture_13
&
capture_14
'
capture_15* 
_
"	capture_0
#	capture_1
$	capture_7
%
capture_13
&
capture_14
'
capture_15* 
_
"	capture_0
#	capture_1
$	capture_7
%
capture_13
&
capture_14
'
capture_15* 
_
"	capture_0
#	capture_1
$	capture_7
%
capture_13
&
capture_14
'
capture_15* 
_
"	capture_0
#	capture_1
$	capture_7
%
capture_13
&
capture_14
'
capture_15* 
_
"	capture_0
#	capture_1
$	capture_7
%
capture_13
&
capture_14
'
capture_15* 
* 
* 
* 
* 
* 
* 
J
0
1
2
3
4
5
6
7
8
9*

0
1
2*
* 
?
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*

Atrace_0
Btrace_1* 

Ctrace_0
Dtrace_1* 
?
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses
K	posterior
	Lprior
	kl_anneal*
?
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses
Sdistance
T
similarity*
?
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses
[	_all_keys
\_input_keys
]gating_keys* 
?
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses
d	_all_keys
e_input_keys
fgating_keys* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
_
"	capture_0
#	capture_1
$	capture_7
%
capture_13
&
capture_14
'
capture_15* 
8
g	variables
h	keras_api
	itotal
	jcount*
H
k	variables
l	keras_api
	mtotal
	ncount
o
_fn_kwargs*
5
0
1
2
3
4
5
6*
 
.0
/1
02
13*
* 
* 
* 
_
"	capture_0
#	capture_1
$	capture_7
%
capture_13
&
capture_14
'
capture_15* 
_
"	capture_0
#	capture_1
$	capture_7
%
capture_13
&
capture_14
'
capture_15* 
_
"	capture_0
#	capture_1
$	capture_7
%
capture_13
&
capture_14
'
capture_15* 
_
"	capture_0
#	capture_1
$	capture_7
%
capture_13
&
capture_14
'
capture_15* 
'
0
1
2
3
4*

0
1
2*
* 
?
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses*
* 
* 
?
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses
loc
untransformed_scale
{
embeddings*
?
|	variables
}trainable_variables
~regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?
_embedding*
'
0
1
2
3
4*
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses*
* 
* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
rho
w*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
tau
	gamma
beta*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

i0
j1*

g	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

m0
n1*

k	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

K0
L1*
* 
* 
* 

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses*
* 
* 
-
?_distribution
?_graph_parents*

0
1*

0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
|	variables
}trainable_variables
~regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
loc
untransformed_scale
?
embeddings*
'
0
1
2
3
4*

S0
T1*
* 
* 
* 

0
1*
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

0
1
2*
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
0
_loc
?_scale
?_graph_parents*
* 

0*

?0*
* 
* 
* 

0
1*

0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
-
?_distribution
?_graph_parents*

0
1*
* 
* 
* 
* 

0
1
2*
* 
* 
* 
* 

_pretransformed_input*
* 

0*
* 
* 
* 
* 
0
_loc
?_scale
?_graph_parents*
* 

_pretransformed_input*
* 
|v
VARIABLE_VALUE Adam/embedding_normal_diag/loc/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE0Adam/embedding_normal_diag/untransformed_scale/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE2Adam/embedding_normal_diag_1/untransformed_scale/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/embedding_normal_diag/loc/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE0Adam/embedding_normal_diag/untransformed_scale/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE2Adam/embedding_normal_diag_1/untransformed_scale/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-embedding_normal_diag/loc/Read/ReadVariableOp=embedding_normal_diag/untransformed_scale/Read/ReadVariableOp?embedding_normal_diag_1/untransformed_scale/Read/ReadVariableOpkl_anneal/Read/ReadVariableOp/embedding_normal_diag_1/loc/Read/ReadVariableOp!minkowski/rho/Read/ReadVariableOpXstochastic_behavior_model/rank_similarity/distance_based/minkowski/w/Read/ReadVariableOp.exponential_similarity/tau/Read/ReadVariableOp0exponential_similarity/gamma/Read/ReadVariableOp/exponential_similarity/beta/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp4Adam/embedding_normal_diag/loc/m/Read/ReadVariableOpDAdam/embedding_normal_diag/untransformed_scale/m/Read/ReadVariableOpFAdam/embedding_normal_diag_1/untransformed_scale/m/Read/ReadVariableOp4Adam/embedding_normal_diag/loc/v/Read/ReadVariableOpDAdam/embedding_normal_diag/untransformed_scale/v/Read/ReadVariableOpFAdam/embedding_normal_diag_1/untransformed_scale/v/Read/ReadVariableOpConst_6*&
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_120344
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_normal_diag/loc)embedding_normal_diag/untransformed_scale+embedding_normal_diag_1/untransformed_scale	kl_annealembedding_normal_diag_1/locminkowski/rhoDstochastic_behavior_model/rank_similarity/distance_based/minkowski/wexponential_similarity/tauexponential_similarity/gammaexponential_similarity/beta	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcount Adam/embedding_normal_diag/loc/m0Adam/embedding_normal_diag/untransformed_scale/m2Adam/embedding_normal_diag_1/untransformed_scale/m Adam/embedding_normal_diag/loc/v0Adam/embedding_normal_diag/untransformed_scale/v2Adam/embedding_normal_diag_1/untransformed_scale/v*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_120429??'
??
?
U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_119142
inputs_2rank1_stimulus_set
inputs_agent_id$
 rank_similarity_gatherv2_indices!
rank_similarity_gatherv2_axise
Srank_similarity_embedding_variational_embedding_normal_diag_embedding_lookup_118857:?
orank_similarity_embedding_variational_embedding_normal_diag_embedding_lookup_1_softplus_readvariableop_resource:x
frank_similarity_embedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_118906:?
?rank_similarity_embedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_1_softplus_readvariableop_resource:M
Crank_similarity_embedding_variational_mul_1_readvariableop_resource: #
rank_similarity_packed_values_1J
@rank_similarity_distance_based_minkowski_readvariableop_resource: Z
Lrank_similarity_distance_based_minkowski_broadcastto_readvariableop_resource:[
Qrank_similarity_distance_based_exponential_similarity_neg_readvariableop_resource: [
Qrank_similarity_distance_based_exponential_similarity_pow_readvariableop_resource: [
Qrank_similarity_distance_based_exponential_similarity_add_readvariableop_resource: &
"rank_similarity_gatherv2_1_indices
rank_similarity_mul_x
rank_similarity_truediv_y
identity

identity_1??Hrank_similarity/distance_based/exponential_similarity/Neg/ReadVariableOp?Hrank_similarity/distance_based/exponential_similarity/Pow/ReadVariableOp?Hrank_similarity/distance_based/exponential_similarity/add/ReadVariableOp?Crank_similarity/distance_based/minkowski/BroadcastTo/ReadVariableOp?7rank_similarity/distance_based/minkowski/ReadVariableOp??rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOp??rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOp??rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOp??rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOp?Lrank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup?frank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOp?_rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup?yrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOp?:rank_similarity/embedding_variational/mul_1/ReadVariableOp??rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp??rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp?
rank_similarity/GatherV2GatherV2inputs_2rank1_stimulus_set rank_similarity_gatherv2_indicesrank_similarity_gatherv2_axis*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:?????????\
rank_similarity/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
rank_similarity/NotEqualNotEqual!rank_similarity/GatherV2:output:0#rank_similarity/NotEqual/y:output:0*
T0*+
_output_shapes
:??????????
Lrank_similarity/embedding_variational/embedding_normal_diag/embedding_lookupResourceGatherSrank_similarity_embedding_variational_embedding_normal_diag_embedding_lookup_118857inputs_2rank1_stimulus_set*
Tindices0*f
_class\
ZXloc:@rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup/118857*/
_output_shapes
:?????????*
dtype0?
Urank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup/IdentityIdentityUrank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup:output:0*
T0*f
_class\
ZXloc:@rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup/118857*/
_output_shapes
:??????????
Wrank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup/Identity_1Identity^rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup/Identity:output:0*
T0*/
_output_shapes
:??????????
frank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOpReadVariableOporank_similarity_embedding_variational_embedding_normal_diag_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
Wrank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/SoftplusSoftplusnrank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
Trank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
Rrank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/addAddV2]rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/add/x:output:0erank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus:activations:0*
T0*
_output_shapes

:?
Srank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/axisConst*e
_class[
YWloc:@rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/add*
_output_shapes
: *
dtype0*
value	B : ?
Nrank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1GatherV2Vrank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/add:z:0inputs_2rank1_stimulus_set\rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*e
_class[
YWloc:@rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/add*/
_output_shapes
:??????????
Wrank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/IdentityIdentityWrank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1:output:0*
T0*/
_output_shapes
:??????????
Vrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Orank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/ShapeShape`rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
Orank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
]rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
_rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
_rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Wrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_sliceStridedSliceXrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/Shape:output:0frank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack:output:0hrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack_1:output:0hrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Qrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/Shape_1Shape`rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/Identity:output:0*
T0*
_output_shapes
:?
Qrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : ?
_rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
arank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
arank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Yrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1StridedSliceZrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/Shape_1:output:0hrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack:output:0jrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack_1:output:0jrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Zrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
\rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
Wrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgsBroadcastArgserank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs/s0_1:output:0`rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice:output:0*
_output_shapes
:?
Yrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs_1BroadcastArgs\rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs:r0:0brank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1:output:0*
_output_shapes
:?
Yrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
Urank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Prank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/concatConcatV2brank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/concat/values_0:output:0^rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs_1:r0:0^rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
crank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
erank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
srank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalYrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/concat:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????*
dtype0?
brank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/mulMul|rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/RandomStandardNormal:output:0nrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/stddev:output:0*
T0*N
_output_shapes<
::8?????????????????????????????????????
^rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normalAddV2frank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/mul:z:0lrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/mean:output:0*
T0*N
_output_shapes<
::8?????????????????????????????????????
Mrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/mulMulbrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal:z:0`rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/Identity:output:0*
T0*E
_output_shapes3
1:/????????????????????????????
Mrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/addAddV2Qrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/mul:z:0`rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup/Identity_1:output:0*
T0*E
_output_shapes3
1:/????????????????????????????
Qrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/Shape_2ShapeQrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/add:z:0*
T0*
_output_shapes
:?
_rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
arank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
arank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Yrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2StridedSliceZrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/Shape_2:output:0hrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack:output:0jrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack_1:output:0jrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Wrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Rrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/concat_1ConcatV2_rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/sample_shape:output:0brank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2:output:0`rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Qrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/ReshapeReshapeQrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/add:z:0[rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/concat_1:output:0*
T0*/
_output_shapes
:??????????
Arank_similarity/embedding_variational/embedding_shared/zeros_like	ZerosLikeinputs_2rank1_stimulus_set*
T0*+
_output_shapes
:??????????
_rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookupResourceGatherfrank_similarity_embedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_118906Erank_similarity/embedding_variational/embedding_shared/zeros_like:y:0*
Tindices0*y
_classo
mkloc:@rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/118906*/
_output_shapes
:?????????*
dtype0?
hrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/IdentityIdentityhrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup:output:0*
T0*y
_classo
mkloc:@rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/118906*/
_output_shapes
:??????????
jrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/Identity_1Identityqrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/Identity:output:0*
T0*/
_output_shapes
:??????????
yrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOpReadVariableOp?rank_similarity_embedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
jrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/SoftplusSoftplus?rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
grank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
erank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/addAddV2prank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add/x:output:0xrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus:activations:0*
T0*
_output_shapes

:?
frank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/axisConst*x
_classn
ljloc:@rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add*
_output_shapes
: *
dtype0*
value	B : ?
arank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1GatherV2irank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add:z:0Erank_similarity/embedding_variational/embedding_shared/zeros_like:y:0orank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*x
_classn
ljloc:@rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add*/
_output_shapes
:??????????
jrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/IdentityIdentityjrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1:output:0*
T0*/
_output_shapes
:??????????
irank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
brank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/ShapeShapesrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
brank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
prank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
rrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
rrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
jrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_sliceStridedSlicekrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape:output:0yrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack:output:0{rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack_1:output:0{rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
drank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape_1Shapesrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Identity:output:0*
T0*
_output_shapes
:?
drank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : ?
rrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
trank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
trank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1StridedSlicemrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape_1:output:0{rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack:output:0}rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack_1:output:0}rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
mrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
orank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
jrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgsBroadcastArgsxrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs/s0_1:output:0srank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice:output:0*
_output_shapes
:?
lrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs_1BroadcastArgsorank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs:r0:0urank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1:output:0*
_output_shapes
:?
lrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
hrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
crank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concatConcatV2urank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat/values_0:output:0qrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs_1:r0:0qrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
vrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
xrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
?rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormallrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????*
dtype0?
urank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/mulMul?rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/RandomStandardNormal:output:0?rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/stddev:output:0*
T0*N
_output_shapes<
::8?????????????????????????????????????
qrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normalAddV2yrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/mul:z:0rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/mean:output:0*
T0*N
_output_shapes<
::8?????????????????????????????????????
`rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/mulMulurank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal:z:0srank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Identity:output:0*
T0*E
_output_shapes3
1:/????????????????????????????
`rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/addAddV2drank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/mul:z:0srank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/Identity_1:output:0*
T0*E
_output_shapes3
1:/????????????????????????????
drank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape_2Shapedrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/add:z:0*
T0*
_output_shapes
:?
rrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
trank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
trank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2StridedSlicemrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape_2:output:0{rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack:output:0}rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack_1:output:0}rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
jrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
erank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat_1ConcatV2rrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/sample_shape:output:0urank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2:output:0srank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
drank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/ReshapeReshapedrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/add:z:0nrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat_1:output:0*
T0*/
_output_shapes
:??????????
=rank_similarity/embedding_variational/reshape/event_shape_outConst*
_output_shapes
: *
dtype0*
valueB ?
<rank_similarity/embedding_variational/reshape/event_shape_inConst*
_output_shapes
:*
dtype0*
valueB"      ?
Jrank_similarity/embedding_variational/SampleIndependentNormal/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
Irank_similarity/embedding_variational/reshapeSampleIndependentNormal/zeroConst*
_output_shapes
: *
dtype0*
value	B : ?
prank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
~rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOpReadVariableOpSrank_similarity_embedding_variational_embedding_normal_diag_embedding_lookup_118857*
_output_shapes

:*
dtype0?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOpReadVariableOporank_similarity_embedding_variational_embedding_normal_diag_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
zrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/SoftplusSoftplus?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
wrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
urank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/addAddV2?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add/x:output:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus:activations:0*
T0*
_output_shapes

:?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"      ?
wrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_sliceStridedSlice?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/shape_as_tensor:output:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack:output:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack_1:output:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"      ?
yrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : ?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1StridedSlice?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/shape_as_tensor_1:output:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack:output:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack_1:output:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgsBroadcastArgs?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs/s0_1:output:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice:output:0*
_output_shapes
:?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs_1BroadcastArgs?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs:r0:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1:output:0*
_output_shapes
:?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
}rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
xrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concatConcatV2?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat/values_0:output:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs_1:r0:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat:output:0*
T0*"
_output_shapes
:*
dtype0?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/mulMul?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/RandomStandardNormal:output:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/stddev:output:0*
T0*"
_output_shapes
:?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normalAddV2?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/mul:z:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/mean:output:0*
T0*"
_output_shapes
:?
urank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/mulMul?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal:z:0yrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add:z:0*
T0*"
_output_shapes
:?
wrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add_1AddV2yrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/mul:z:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOp:value:0*
T0*"
_output_shapes
:?
rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
yrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReshapeReshape{rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add_1:z:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Reshape/shape:output:0*
T0*"
_output_shapes
:?
qrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
krank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/ReshapeReshape?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Reshape:output:0zrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Reshape/shape:output:0*
T0*"
_output_shapes
:?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOpReadVariableOporank_similarity_embedding_variational_embedding_normal_diag_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
~rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/SoftplusSoftplus?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
{rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
yrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/addAddV2?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add/x:output:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus:activations:0*
T0*
_output_shapes

:?
}rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/truedivRealDivtrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Reshape:output:0}rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add:z:0*
T0*"
_output_shapes
:?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOpReadVariableOpSrank_similarity_embedding_variational_embedding_normal_diag_embedding_lookup_118857*
_output_shapes

:*
dtype0?
rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/truediv_1RealDiv?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOp:value:0}rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add:z:0*
T0*
_output_shapes

:?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/SquaredDifferenceSquaredDifference?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/truediv:z:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/truediv_1:z:0*
T0*"
_output_shapes
:?
{rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
yrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/mulMul?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/mul/x:output:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/SquaredDifference:z:0*
T0*"
_output_shapes
:?
{rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??k??
yrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/LogLog}rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add:z:0*
T0*
_output_shapes

:?
{rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add_1AddV2?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Const:output:0}rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Log:y:0*
T0*
_output_shapes

:?
yrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/subSub}rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/mul:z:0rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add_1:z:0*
T0*"
_output_shapes
:?
{rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"?????????
irank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/SumSum}rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/sub:z:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Sum/reduction_indices:output:0*
T0*
_output_shapes
:?
jrank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/inverse/output_shapeConst*
_output_shapes
:*
dtype0*)
value B"               ?
erank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/inverse/ReshapeReshapetrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Reshape:output:0srank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/inverse/output_shape:output:0*
T0**
_output_shapes
:?
~rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/event_shape_tensor/ConstConst*
_output_shapes
:*
dtype0*%
valueB"            ?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/event_shape_tensor/event_shapeIdentity?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/event_shape_tensor/Const:output:0*
T0*
_output_shapes
:?
yrank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_event_shape_tensor/IdentityIdentity?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/event_shape_tensor/event_shape:output:0*
T0*
_output_shapes
:?
}rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_event_shape_tensor/output_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
|rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"               ?
vrank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/ReshapeReshapenrank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/inverse/Reshape:output:0?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Reshape/shape:output:0*
T0**
_output_shapes
:?
}rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                ?
xrank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/transpose	Transposerank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Reshape:output:0?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/transpose/perm:output:0*
T0**
_output_shapes
:?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOpReadVariableOp?rank_similarity_embedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/SoftplusSoftplus?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/addAddV2?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add/x:output:0?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus:activations:0*
T0*
_output_shapes

:?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/truedivRealDiv|rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/transpose:y:0?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add:z:0*
T0**
_output_shapes
:?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOpReadVariableOpfrank_similarity_embedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_118906*
_output_shapes

:*
dtype0?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/truediv_1RealDiv?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp:value:0?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add:z:0*
T0*
_output_shapes

:?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/SquaredDifferenceSquaredDifference?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/truediv:z:0?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/truediv_1:z:0*
T0**
_output_shapes
:?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/mulMul?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/mul/x:output:0?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/SquaredDifference:z:0*
T0**
_output_shapes
:?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??k??
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/LogLog?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add:z:0*
T0*
_output_shapes

:?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add_1AddV2?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Const:output:0?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Log:y:0*
T0*
_output_shapes

:?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/subSub?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/mul:z:0?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add_1:z:0*
T0**
_output_shapes
:?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"?????????
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/SumSum?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/sub:z:0?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Sum/reduction_indices:output:0*
T0*"
_output_shapes
:?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/BroadcastTo/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
zrank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/BroadcastToBroadcastTo?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Sum:output:0?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/BroadcastTo/shape:output:0*
T0*"
_output_shapes
:?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
rrank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/SumSum?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/BroadcastTo:output:0?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Sum/reduction_indices:output:0*
T0*
_output_shapes
:?
trank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
rrank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/NegNeg}rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/zeros:output:0*
T0*
_output_shapes
: ?
trank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Neg_1Negvrank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Neg:y:0*
T0*
_output_shapes
: ?
srank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/onesConst*
_output_shapes

:*
dtype0*
valueB*  ???
rrank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/mulMul|rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/ones:output:0xrank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Neg_1:y:0*
T0*
_output_shapes

:?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"?????????
rrank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/SumSumvrank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/mul:z:0?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Sum/reduction_indices:output:0*
T0*
_output_shapes
: ?
Qrank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/subSub{rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Sum:output:0{rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Sum:output:0*
T0*
_output_shapes
:?
)rank_similarity/embedding_variational/subSubrrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Sum:output:0Urank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/sub:z:0*
T0*
_output_shapes
:u
+rank_similarity/embedding_variational/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
*rank_similarity/embedding_variational/MeanMean-rank_similarity/embedding_variational/sub:z:04rank_similarity/embedding_variational/Const:output:0*
T0*
_output_shapes
: p
+rank_similarity/embedding_variational/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *x?V9?
)rank_similarity/embedding_variational/mulMul3rank_similarity/embedding_variational/Mean:output:04rank_similarity/embedding_variational/mul/y:output:0*
T0*
_output_shapes
: ?
:rank_similarity/embedding_variational/mul_1/ReadVariableOpReadVariableOpCrank_similarity_embedding_variational_mul_1_readvariableop_resource*
_output_shapes
: *
dtype0?
+rank_similarity/embedding_variational/mul_1Mul-rank_similarity/embedding_variational/mul:z:0Brank_similarity/embedding_variational/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
: Z
rank_similarity/packed/0Const*
_output_shapes
: *
dtype0*
value	B :?
rank_similarity/packedPack!rank_similarity/packed/0:output:0rank_similarity_packed_values_1*
N*
T0*
_output_shapes
:a
rank_similarity/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
rank_similarity/splitSplitVZrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/Reshape:output:0rank_similarity/packed:output:0(rank_similarity/split/split_dim:output:0*
T0*

Tlen0*J
_output_shapes8
6:?????????:?????????*
	num_split?
,rank_similarity/distance_based/minkowski/subSubrank_similarity/split:output:0rank_similarity/split:output:1*
T0*/
_output_shapes
:??????????
.rank_similarity/distance_based/minkowski/ShapeShape0rank_similarity/distance_based/minkowski/sub:z:0*
T0*
_output_shapes
:?
<rank_similarity/distance_based/minkowski/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
>rank_similarity/distance_based/minkowski/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
>rank_similarity/distance_based/minkowski/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
6rank_similarity/distance_based/minkowski/strided_sliceStridedSlice7rank_similarity/distance_based/minkowski/Shape:output:0Erank_similarity/distance_based/minkowski/strided_slice/stack:output:0Grank_similarity/distance_based/minkowski/strided_slice/stack_1:output:0Grank_similarity/distance_based/minkowski/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:x
3rank_similarity/distance_based/minkowski/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
-rank_similarity/distance_based/minkowski/onesFill?rank_similarity/distance_based/minkowski/strided_slice:output:0<rank_similarity/distance_based/minkowski/ones/Const:output:0*
T0*+
_output_shapes
:??????????
7rank_similarity/distance_based/minkowski/ReadVariableOpReadVariableOp@rank_similarity_distance_based_minkowski_readvariableop_resource*
_output_shapes
: *
dtype0?
,rank_similarity/distance_based/minkowski/mulMul?rank_similarity/distance_based/minkowski/ReadVariableOp:value:06rank_similarity/distance_based/minkowski/ones:output:0*
T0*+
_output_shapes
:??????????
Crank_similarity/distance_based/minkowski/BroadcastTo/ReadVariableOpReadVariableOpLrank_similarity_distance_based_minkowski_broadcastto_readvariableop_resource*
_output_shapes
:*
dtype0?
4rank_similarity/distance_based/minkowski/BroadcastToBroadcastToKrank_similarity/distance_based/minkowski/BroadcastTo/ReadVariableOp:value:07rank_similarity/distance_based/minkowski/Shape:output:0*
T0*/
_output_shapes
:??????????
,rank_similarity/distance_based/minkowski/AbsAbs0rank_similarity/distance_based/minkowski/sub:z:0*
T0*/
_output_shapes
:??????????
7rank_similarity/distance_based/minkowski/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
3rank_similarity/distance_based/minkowski/ExpandDims
ExpandDims0rank_similarity/distance_based/minkowski/mul:z:0@rank_similarity/distance_based/minkowski/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
,rank_similarity/distance_based/minkowski/PowPow0rank_similarity/distance_based/minkowski/Abs:y:0<rank_similarity/distance_based/minkowski/ExpandDims:output:0*
T0*/
_output_shapes
:??????????
.rank_similarity/distance_based/minkowski/Mul_1Mul0rank_similarity/distance_based/minkowski/Pow:z:0=rank_similarity/distance_based/minkowski/BroadcastTo:output:0*
T0*/
_output_shapes
:??????????
>rank_similarity/distance_based/minkowski/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
,rank_similarity/distance_based/minkowski/SumSum2rank_similarity/distance_based/minkowski/Mul_1:z:0Grank_similarity/distance_based/minkowski/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(w
2rank_similarity/distance_based/minkowski/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
0rank_similarity/distance_based/minkowski/truedivRealDiv;rank_similarity/distance_based/minkowski/truediv/x:output:0<rank_similarity/distance_based/minkowski/ExpandDims:output:0*
T0*/
_output_shapes
:??????????
.rank_similarity/distance_based/minkowski/Pow_1Pow5rank_similarity/distance_based/minkowski/Sum:output:04rank_similarity/distance_based/minkowski/truediv:z:0*
T0*/
_output_shapes
:??????????
1rank_similarity/distance_based/minkowski/IdentityIdentity2rank_similarity/distance_based/minkowski/Pow_1:z:0*
T0*/
_output_shapes
:??????????
2rank_similarity/distance_based/minkowski/IdentityN	IdentityN2rank_similarity/distance_based/minkowski/Pow_1:z:00rank_similarity/distance_based/minkowski/sub:z:0=rank_similarity/distance_based/minkowski/BroadcastTo:output:00rank_similarity/distance_based/minkowski/mul:z:0*
T
2*,
_gradient_op_typeCustomGradient-119079*|
_output_shapesj
h:?????????:?????????:?????????:??????????
0rank_similarity/distance_based/minkowski/SqueezeSqueeze;rank_similarity/distance_based/minkowski/IdentityN:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

??????????
Hrank_similarity/distance_based/exponential_similarity/Neg/ReadVariableOpReadVariableOpQrank_similarity_distance_based_exponential_similarity_neg_readvariableop_resource*
_output_shapes
: *
dtype0?
9rank_similarity/distance_based/exponential_similarity/NegNegPrank_similarity/distance_based/exponential_similarity/Neg/ReadVariableOp:value:0*
T0*
_output_shapes
: ?
Hrank_similarity/distance_based/exponential_similarity/Pow/ReadVariableOpReadVariableOpQrank_similarity_distance_based_exponential_similarity_pow_readvariableop_resource*
_output_shapes
: *
dtype0?
9rank_similarity/distance_based/exponential_similarity/PowPow9rank_similarity/distance_based/minkowski/Squeeze:output:0Prank_similarity/distance_based/exponential_similarity/Pow/ReadVariableOp:value:0*
T0*+
_output_shapes
:??????????
9rank_similarity/distance_based/exponential_similarity/mulMul=rank_similarity/distance_based/exponential_similarity/Neg:y:0=rank_similarity/distance_based/exponential_similarity/Pow:z:0*
T0*+
_output_shapes
:??????????
9rank_similarity/distance_based/exponential_similarity/ExpExp=rank_similarity/distance_based/exponential_similarity/mul:z:0*
T0*+
_output_shapes
:??????????
Hrank_similarity/distance_based/exponential_similarity/add/ReadVariableOpReadVariableOpQrank_similarity_distance_based_exponential_similarity_add_readvariableop_resource*
_output_shapes
: *
dtype0?
9rank_similarity/distance_based/exponential_similarity/addAddV2=rank_similarity/distance_based/exponential_similarity/Exp:y:0Prank_similarity/distance_based/exponential_similarity/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????
rank_similarity/CastCastrank_similarity/NotEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:??????????
,rank_similarity/rank_sim_zero_out_nonpresentMul=rank_similarity/distance_based/exponential_similarity/add:z:0rank_similarity/Cast:y:0*
T0*+
_output_shapes
:??????????
rank_similarity/GatherV2_1GatherV20rank_similarity/rank_sim_zero_out_nonpresent:z:0"rank_similarity_gatherv2_1_indicesrank_similarity_gatherv2_axis*
Taxis0*
Tindices0*
Tparams0*/
_output_shapes
:?????????`
rank_similarity/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
rank_similarity/ExpandDims
ExpandDimsrank_similarity/Cast:y:0'rank_similarity/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????W
rank_similarity/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
rank_similarity/GatherV2_2GatherV2#rank_similarity/ExpandDims:output:0rank_similarity/Const:output:0rank_similarity_gatherv2_axis*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:?????????]
rank_similarity/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B :?
rank_similarity/CumsumCumsum#rank_similarity/GatherV2_1:output:0$rank_similarity/Cumsum/axis:output:0*
T0*/
_output_shapes
:?????????*
reverse(^
rank_similarity/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
rank_similarity/MaximumMaximum#rank_similarity/GatherV2_1:output:0"rank_similarity/Maximum/y:output:0*
T0*/
_output_shapes
:?????????`
rank_similarity/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
rank_similarity/Maximum_1Maximumrank_similarity/Cumsum:out:0$rank_similarity/Maximum_1/y:output:0*
T0*/
_output_shapes
:?????????q
rank_similarity/LogLogrank_similarity/Maximum:z:0*
T0*/
_output_shapes
:?????????u
rank_similarity/Log_1Logrank_similarity/Maximum_1:z:0*
T0*/
_output_shapes
:??????????
rank_similarity/subSubrank_similarity/Log:y:0rank_similarity/Log_1:y:0*
T0*/
_output_shapes
:??????????
rank_similarity/mulMulrank_similarity_mul_xrank_similarity/sub:z:0*
T0*/
_output_shapes
:?????????g
%rank_similarity/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
rank_similarity/SumSumrank_similarity/mul:z:0.rank_similarity/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????n
rank_similarity/ExpExprank_similarity/Sum:output:0*
T0*+
_output_shapes
:??????????
rank_similarity/mul_1Mul#rank_similarity/GatherV2_2:output:0rank_similarity/Exp:y:0*
T0*+
_output_shapes
:?????????i
'rank_similarity/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
rank_similarity/Sum_1Sumrank_similarity/mul_1:z:00rank_similarity/Sum_1/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(\
rank_similarity/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
rank_similarity/EqualEqualrank_similarity/Sum_1:output:0 rank_similarity/Equal/y:output:0*
T0*+
_output_shapes
:?????????~
rank_similarity/Cast_1Castrank_similarity/Equal:z:0*

DstT0*

SrcT0
*+
_output_shapes
:??????????
rank_similarity/truedivRealDivrank_similarity/Cast_1:y:0rank_similarity_truediv_y*
T0*+
_output_shapes
:??????????
rank_similarity/addAddV2rank_similarity/mul_1:z:0rank_similarity/truediv:z:0*
T0*+
_output_shapes
:??????????
rank_similarity/add_1AddV2rank_similarity/Sum_1:output:0rank_similarity/Cast_1:y:0*
T0*+
_output_shapes
:??????????
rank_similarity/truediv_1RealDivrank_similarity/add:z:0rank_similarity/add_1:z:0*
T0*+
_output_shapes
:?????????p
IdentityIdentityrank_similarity/truediv_1:z:0^NoOp*
T0*+
_output_shapes
:?????????o

Identity_1Identity/rank_similarity/embedding_variational/mul_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOpI^rank_similarity/distance_based/exponential_similarity/Neg/ReadVariableOpI^rank_similarity/distance_based/exponential_similarity/Pow/ReadVariableOpI^rank_similarity/distance_based/exponential_similarity/add/ReadVariableOpD^rank_similarity/distance_based/minkowski/BroadcastTo/ReadVariableOp8^rank_similarity/distance_based/minkowski/ReadVariableOp?^rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOp?^rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOp?^rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOp?^rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOpM^rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookupg^rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOp`^rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookupz^rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOp;^rank_similarity/embedding_variational/mul_1/ReadVariableOp?^rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp?^rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:?????????:?????????:: : : : : : : : : : : : ::: 2?
Hrank_similarity/distance_based/exponential_similarity/Neg/ReadVariableOpHrank_similarity/distance_based/exponential_similarity/Neg/ReadVariableOp2?
Hrank_similarity/distance_based/exponential_similarity/Pow/ReadVariableOpHrank_similarity/distance_based/exponential_similarity/Pow/ReadVariableOp2?
Hrank_similarity/distance_based/exponential_similarity/add/ReadVariableOpHrank_similarity/distance_based/exponential_similarity/add/ReadVariableOp2?
Crank_similarity/distance_based/minkowski/BroadcastTo/ReadVariableOpCrank_similarity/distance_based/minkowski/BroadcastTo/ReadVariableOp2r
7rank_similarity/distance_based/minkowski/ReadVariableOp7rank_similarity/distance_based/minkowski/ReadVariableOp2?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOp?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOp2?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOp?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOp2?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOp?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOp2?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOp?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOp2?
Lrank_similarity/embedding_variational/embedding_normal_diag/embedding_lookupLrank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup2?
frank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOpfrank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOp2?
_rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup2?
yrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOpyrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOp2x
:rank_similarity/embedding_variational/mul_1/ReadVariableOp:rank_similarity/embedding_variational/mul_1/ReadVariableOp2?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp2?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp:g c
+
_output_shapes
:?????????
4
_user_specified_nameinputs/2rank1/stimulus_set:\X
+
_output_shapes
:?????????
)
_user_specified_nameinputs/agent_id: 

_output_shapes
::

_output_shapes
: :	

_output_shapes
: :$ 

_output_shapes

::,(
&
_output_shapes
::

_output_shapes
: 
??
?
K__inference_rank_similarity_layer_call_and_return_conditional_losses_117747

inputs
inputs_1
gatherv2_indices
gatherv2_axisU
Cembedding_variational_embedding_normal_diag_embedding_lookup_117462:q
_embedding_variational_embedding_normal_diag_embedding_lookup_1_softplus_readvariableop_resource:h
Vembedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_117511:?
rembedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_1_softplus_readvariableop_resource:=
3embedding_variational_mul_1_readvariableop_resource: 
packed_values_1:
0distance_based_minkowski_readvariableop_resource: J
<distance_based_minkowski_broadcastto_readvariableop_resource:K
Adistance_based_exponential_similarity_neg_readvariableop_resource: K
Adistance_based_exponential_similarity_pow_readvariableop_resource: K
Adistance_based_exponential_similarity_add_readvariableop_resource: 
gatherv2_1_indices	
mul_x
	truediv_y
identity

identity_1??8distance_based/exponential_similarity/Neg/ReadVariableOp?8distance_based/exponential_similarity/Pow/ReadVariableOp?8distance_based/exponential_similarity/add/ReadVariableOp?3distance_based/minkowski/BroadcastTo/ReadVariableOp?'distance_based/minkowski/ReadVariableOp?tembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOp?}embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOp?pembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOp?yembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOp?<embedding_variational/embedding_normal_diag/embedding_lookup?Vembedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOp?Oembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup?iembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOp?*embedding_variational/mul_1/ReadVariableOp??embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp??embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp?
GatherV2GatherV2inputsgatherv2_indicesgatherv2_axis*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:?????????L

NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : r
NotEqualNotEqualGatherV2:output:0NotEqual/y:output:0*
T0*+
_output_shapes
:??????????
<embedding_variational/embedding_normal_diag/embedding_lookupResourceGatherCembedding_variational_embedding_normal_diag_embedding_lookup_117462inputs*
Tindices0*V
_classL
JHloc:@embedding_variational/embedding_normal_diag/embedding_lookup/117462*/
_output_shapes
:?????????*
dtype0?
Eembedding_variational/embedding_normal_diag/embedding_lookup/IdentityIdentityEembedding_variational/embedding_normal_diag/embedding_lookup:output:0*
T0*V
_classL
JHloc:@embedding_variational/embedding_normal_diag/embedding_lookup/117462*/
_output_shapes
:??????????
Gembedding_variational/embedding_normal_diag/embedding_lookup/Identity_1IdentityNembedding_variational/embedding_normal_diag/embedding_lookup/Identity:output:0*
T0*/
_output_shapes
:??????????
Vembedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOpReadVariableOp_embedding_variational_embedding_normal_diag_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
Gembedding_variational/embedding_normal_diag/embedding_lookup_1/SoftplusSoftplus^embedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
Dembedding_variational/embedding_normal_diag/embedding_lookup_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
Bembedding_variational/embedding_normal_diag/embedding_lookup_1/addAddV2Membedding_variational/embedding_normal_diag/embedding_lookup_1/add/x:output:0Uembedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus:activations:0*
T0*
_output_shapes

:?
Cembedding_variational/embedding_normal_diag/embedding_lookup_1/axisConst*U
_classK
IGloc:@embedding_variational/embedding_normal_diag/embedding_lookup_1/add*
_output_shapes
: *
dtype0*
value	B : ?
>embedding_variational/embedding_normal_diag/embedding_lookup_1GatherV2Fembedding_variational/embedding_normal_diag/embedding_lookup_1/add:z:0inputsLembedding_variational/embedding_normal_diag/embedding_lookup_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*U
_classK
IGloc:@embedding_variational/embedding_normal_diag/embedding_lookup_1/add*/
_output_shapes
:??????????
Gembedding_variational/embedding_normal_diag/embedding_lookup_1/IdentityIdentityGembedding_variational/embedding_normal_diag/embedding_lookup_1:output:0*
T0*/
_output_shapes
:??????????
Fembedding_variational/embedding_normal_diag/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
?embedding_variational/embedding_normal_diag/Normal/sample/ShapeShapePembedding_variational/embedding_normal_diag/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
?embedding_variational/embedding_normal_diag/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
Membedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Oembedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Oembedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Gembedding_variational/embedding_normal_diag/Normal/sample/strided_sliceStridedSliceHembedding_variational/embedding_normal_diag/Normal/sample/Shape:output:0Vembedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack:output:0Xembedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack_1:output:0Xembedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Aembedding_variational/embedding_normal_diag/Normal/sample/Shape_1ShapePembedding_variational/embedding_normal_diag/embedding_lookup_1/Identity:output:0*
T0*
_output_shapes
:?
Aembedding_variational/embedding_normal_diag/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : ?
Oembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Qembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Qembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Iembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1StridedSliceJembedding_variational/embedding_normal_diag/Normal/sample/Shape_1:output:0Xembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack:output:0Zembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack_1:output:0Zembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Jembedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
Lembedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
Gembedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgsBroadcastArgsUembedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs/s0_1:output:0Pembedding_variational/embedding_normal_diag/Normal/sample/strided_slice:output:0*
_output_shapes
:?
Iembedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs_1BroadcastArgsLembedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs:r0:0Rembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1:output:0*
_output_shapes
:?
Iembedding_variational/embedding_normal_diag/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
Eembedding_variational/embedding_normal_diag/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
@embedding_variational/embedding_normal_diag/Normal/sample/concatConcatV2Rembedding_variational/embedding_normal_diag/Normal/sample/concat/values_0:output:0Nembedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs_1:r0:0Nembedding_variational/embedding_normal_diag/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Sembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
Uembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
cembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalIembedding_variational/embedding_normal_diag/Normal/sample/concat:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????*
dtype0?
Rembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/mulMullembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/RandomStandardNormal:output:0^embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/stddev:output:0*
T0*N
_output_shapes<
::8?????????????????????????????????????
Nembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normalAddV2Vembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/mul:z:0\embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/mean:output:0*
T0*N
_output_shapes<
::8?????????????????????????????????????
=embedding_variational/embedding_normal_diag/Normal/sample/mulMulRembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal:z:0Pembedding_variational/embedding_normal_diag/embedding_lookup_1/Identity:output:0*
T0*E
_output_shapes3
1:/????????????????????????????
=embedding_variational/embedding_normal_diag/Normal/sample/addAddV2Aembedding_variational/embedding_normal_diag/Normal/sample/mul:z:0Pembedding_variational/embedding_normal_diag/embedding_lookup/Identity_1:output:0*
T0*E
_output_shapes3
1:/????????????????????????????
Aembedding_variational/embedding_normal_diag/Normal/sample/Shape_2ShapeAembedding_variational/embedding_normal_diag/Normal/sample/add:z:0*
T0*
_output_shapes
:?
Oembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Qembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Qembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Iembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2StridedSliceJembedding_variational/embedding_normal_diag/Normal/sample/Shape_2:output:0Xembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack:output:0Zembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack_1:output:0Zembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Gembedding_variational/embedding_normal_diag/Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Bembedding_variational/embedding_normal_diag/Normal/sample/concat_1ConcatV2Oembedding_variational/embedding_normal_diag/Normal/sample/sample_shape:output:0Rembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2:output:0Pembedding_variational/embedding_normal_diag/Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Aembedding_variational/embedding_normal_diag/Normal/sample/ReshapeReshapeAembedding_variational/embedding_normal_diag/Normal/sample/add:z:0Kembedding_variational/embedding_normal_diag/Normal/sample/concat_1:output:0*
T0*/
_output_shapes
:?????????|
1embedding_variational/embedding_shared/zeros_like	ZerosLikeinputs*
T0*+
_output_shapes
:??????????
Oembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookupResourceGatherVembedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_1175115embedding_variational/embedding_shared/zeros_like:y:0*
Tindices0*i
_class_
][loc:@embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/117511*/
_output_shapes
:?????????*
dtype0?
Xembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/IdentityIdentityXembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup:output:0*
T0*i
_class_
][loc:@embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/117511*/
_output_shapes
:??????????
Zembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/Identity_1Identityaembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/Identity:output:0*
T0*/
_output_shapes
:??????????
iembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOpReadVariableOprembedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
Zembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/SoftplusSoftplusqembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
Wembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
Uembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/addAddV2`embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add/x:output:0hembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus:activations:0*
T0*
_output_shapes

:?
Vembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/axisConst*h
_class^
\Zloc:@embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add*
_output_shapes
: *
dtype0*
value	B : ?
Qembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1GatherV2Yembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add:z:05embedding_variational/embedding_shared/zeros_like:y:0_embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*h
_class^
\Zloc:@embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add*/
_output_shapes
:??????????
Zembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/IdentityIdentityZembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1:output:0*
T0*/
_output_shapes
:??????????
Yembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Rembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/ShapeShapecembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
Rembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
`embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
bembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
bembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Zembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_sliceStridedSlice[embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape:output:0iembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack:output:0kembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack_1:output:0kembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Tembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape_1Shapecembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Identity:output:0*
T0*
_output_shapes
:?
Tembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : ?
bembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
dembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
dembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1StridedSlice]embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape_1:output:0kembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack:output:0membedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack_1:output:0membedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
]embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
_embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
Zembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgsBroadcastArgshembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs/s0_1:output:0cembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice:output:0*
_output_shapes
:?
\embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs_1BroadcastArgs_embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs:r0:0eembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1:output:0*
_output_shapes
:?
\embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
Xembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Sembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concatConcatV2eembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat/values_0:output:0aembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs_1:r0:0aembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
fembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
hembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
vembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal\embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????*
dtype0?
eembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/mulMulembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/RandomStandardNormal:output:0qembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/stddev:output:0*
T0*N
_output_shapes<
::8?????????????????????????????????????
aembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normalAddV2iembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/mul:z:0oembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/mean:output:0*
T0*N
_output_shapes<
::8?????????????????????????????????????
Pembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/mulMuleembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal:z:0cembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Identity:output:0*
T0*E
_output_shapes3
1:/????????????????????????????
Pembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/addAddV2Tembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/mul:z:0cembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/Identity_1:output:0*
T0*E
_output_shapes3
1:/????????????????????????????
Tembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape_2ShapeTembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/add:z:0*
T0*
_output_shapes
:?
bembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
dembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2StridedSlice]embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape_2:output:0kembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack:output:0membedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack_1:output:0membedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Zembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Uembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat_1ConcatV2bembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/sample_shape:output:0eembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2:output:0cembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Tembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/ReshapeReshapeTembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/add:z:0^embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat_1:output:0*
T0*/
_output_shapes
:?????????p
-embedding_variational/reshape/event_shape_outConst*
_output_shapes
: *
dtype0*
valueB }
,embedding_variational/reshape/event_shape_inConst*
_output_shapes
:*
dtype0*
valueB"      ?
:embedding_variational/SampleIndependentNormal/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB"      {
9embedding_variational/reshapeSampleIndependentNormal/zeroConst*
_output_shapes
: *
dtype0*
value	B : ?
`embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
qembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
nembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
pembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOpReadVariableOpCembedding_variational_embedding_normal_diag_embedding_lookup_117462*
_output_shapes

:*
dtype0?
yembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOpReadVariableOp_embedding_variational_embedding_normal_diag_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
jembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/SoftplusSoftplus?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
gembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
eembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/addAddV2pembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add/x:output:0xembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus:activations:0*
T0*
_output_shapes

:?
qembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"      ?
gembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
uembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
wembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
wembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
oembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_sliceStridedSlicezembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/shape_as_tensor:output:0~embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack:output:0?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack_1:output:0?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
sembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"      ?
iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : ?
wembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
yembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
yembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
qembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1StridedSlice|embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/shape_as_tensor_1:output:0?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack:output:0?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack_1:output:0?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
rembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
tembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
oembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgsBroadcastArgs}embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs/s0_1:output:0xembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice:output:0*
_output_shapes
:?
qembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs_1BroadcastArgstembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs:r0:0zembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1:output:0*
_output_shapes
:?
qembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
membedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
hembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concatConcatV2zembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat/values_0:output:0vembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs_1:r0:0vembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
{embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
}embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalqembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat:output:0*
T0*"
_output_shapes
:*
dtype0?
zembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/mulMul?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/RandomStandardNormal:output:0?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/stddev:output:0*
T0*"
_output_shapes
:?
vembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normalAddV2~embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/mul:z:0?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/mean:output:0*
T0*"
_output_shapes
:?
eembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/mulMulzembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal:z:0iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add:z:0*
T0*"
_output_shapes
:?
gembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add_1AddV2iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/mul:z:0xembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOp:value:0*
T0*"
_output_shapes
:?
oembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReshapeReshapekembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add_1:z:0xembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Reshape/shape:output:0*
T0*"
_output_shapes
:?
aembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
[embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/ReshapeReshaperembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Reshape:output:0jembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Reshape/shape:output:0*
T0*"
_output_shapes
:?
}embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOpReadVariableOp_embedding_variational_embedding_normal_diag_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
nembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/SoftplusSoftplus?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
kembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/addAddV2tembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add/x:output:0|embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus:activations:0*
T0*
_output_shapes

:?
membedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/truedivRealDivdembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Reshape:output:0membedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add:z:0*
T0*"
_output_shapes
:?
tembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOpReadVariableOpCembedding_variational_embedding_normal_diag_embedding_lookup_117462*
_output_shapes

:*
dtype0?
oembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/truediv_1RealDiv|embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOp:value:0membedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add:z:0*
T0*
_output_shapes

:?
wembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/SquaredDifferenceSquaredDifferenceqembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/truediv:z:0sembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/truediv_1:z:0*
T0*"
_output_shapes
:?
kembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/mulMultembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/mul/x:output:0{embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/SquaredDifference:z:0*
T0*"
_output_shapes
:?
kembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??k??
iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/LogLogmembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add:z:0*
T0*
_output_shapes

:?
kembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add_1AddV2tembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Const:output:0membedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Log:y:0*
T0*
_output_shapes

:?
iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/subSubmembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/mul:z:0oembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add_1:z:0*
T0*"
_output_shapes
:?
kembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"?????????
Yembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/SumSummembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/sub:z:0tembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Sum/reduction_indices:output:0*
T0*
_output_shapes
:?
Zembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/inverse/output_shapeConst*
_output_shapes
:*
dtype0*)
value B"               ?
Uembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/inverse/ReshapeReshapedembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Reshape:output:0cembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/inverse/output_shape:output:0*
T0**
_output_shapes
:?
nembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/event_shape_tensor/ConstConst*
_output_shapes
:*
dtype0*%
valueB"            ?
tembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/event_shape_tensor/event_shapeIdentitywembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/event_shape_tensor/Const:output:0*
T0*
_output_shapes
:?
iembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_event_shape_tensor/IdentityIdentity}embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/event_shape_tensor/event_shape:output:0*
T0*
_output_shapes
:?
membedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_event_shape_tensor/output_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
lembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"               ?
fembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/ReshapeReshape^embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/inverse/Reshape:output:0uembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Reshape/shape:output:0*
T0**
_output_shapes
:?
membedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                ?
hembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/transpose	Transposeoembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Reshape:output:0vembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/transpose/perm:output:0*
T0**
_output_shapes
:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOpReadVariableOprembedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/SoftplusSoftplus?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/addAddV2?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add/x:output:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus:activations:0*
T0*
_output_shapes

:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/truedivRealDivlembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/transpose:y:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add:z:0*
T0**
_output_shapes
:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOpReadVariableOpVembedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_117511*
_output_shapes

:*
dtype0?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/truediv_1RealDiv?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp:value:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add:z:0*
T0*
_output_shapes

:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/SquaredDifferenceSquaredDifference?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/truediv:z:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/truediv_1:z:0*
T0**
_output_shapes
:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/mulMul?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/mul/x:output:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/SquaredDifference:z:0*
T0**
_output_shapes
:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??k??
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/LogLog?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add:z:0*
T0*
_output_shapes

:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add_1AddV2?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Const:output:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Log:y:0*
T0*
_output_shapes

:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/subSub?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/mul:z:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add_1:z:0*
T0**
_output_shapes
:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"?????????
}embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/SumSum?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/sub:z:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Sum/reduction_indices:output:0*
T0*"
_output_shapes
:?
pembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/BroadcastTo/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
jembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/BroadcastToBroadcastTo?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Sum:output:0yembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/BroadcastTo/shape:output:0*
T0*"
_output_shapes
:?
tembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
bembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/SumSumsembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/BroadcastTo:output:0}embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Sum/reduction_indices:output:0*
T0*
_output_shapes
:?
dembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
bembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/NegNegmembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/zeros:output:0*
T0*
_output_shapes
: ?
dembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Neg_1Negfembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Neg:y:0*
T0*
_output_shapes
: ?
cembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/onesConst*
_output_shapes

:*
dtype0*
valueB*  ???
bembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/mulMullembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/ones:output:0hembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Neg_1:y:0*
T0*
_output_shapes

:?
tembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"?????????
bembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/SumSumfembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/mul:z:0}embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Sum/reduction_indices:output:0*
T0*
_output_shapes
: ?
Aembedding_variational/reshapeSampleIndependentNormal/log_prob/subSubkembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Sum:output:0kembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Sum:output:0*
T0*
_output_shapes
:?
embedding_variational/subSubbembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Sum:output:0Eembedding_variational/reshapeSampleIndependentNormal/log_prob/sub:z:0*
T0*
_output_shapes
:e
embedding_variational/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
embedding_variational/MeanMeanembedding_variational/sub:z:0$embedding_variational/Const:output:0*
T0*
_output_shapes
: `
embedding_variational/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *x?V9?
embedding_variational/mulMul#embedding_variational/Mean:output:0$embedding_variational/mul/y:output:0*
T0*
_output_shapes
: ?
*embedding_variational/mul_1/ReadVariableOpReadVariableOp3embedding_variational_mul_1_readvariableop_resource*
_output_shapes
: *
dtype0?
embedding_variational/mul_1Mulembedding_variational/mul:z:02embedding_variational/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
: J
packed/0Const*
_output_shapes
: *
dtype0*
value	B :`
packedPackpacked/0:output:0packed_values_1*
N*
T0*
_output_shapes
:Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitVJembedding_variational/embedding_normal_diag/Normal/sample/Reshape:output:0packed:output:0split/split_dim:output:0*
T0*

Tlen0*J
_output_shapes8
6:?????????:?????????*
	num_split}
distance_based/minkowski/subSubsplit:output:0split:output:1*
T0*/
_output_shapes
:?????????n
distance_based/minkowski/ShapeShape distance_based/minkowski/sub:z:0*
T0*
_output_shapes
:v
,distance_based/minkowski/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
.distance_based/minkowski/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????x
.distance_based/minkowski/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&distance_based/minkowski/strided_sliceStridedSlice'distance_based/minkowski/Shape:output:05distance_based/minkowski/strided_slice/stack:output:07distance_based/minkowski/strided_slice/stack_1:output:07distance_based/minkowski/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:h
#distance_based/minkowski/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
distance_based/minkowski/onesFill/distance_based/minkowski/strided_slice:output:0,distance_based/minkowski/ones/Const:output:0*
T0*+
_output_shapes
:??????????
'distance_based/minkowski/ReadVariableOpReadVariableOp0distance_based_minkowski_readvariableop_resource*
_output_shapes
: *
dtype0?
distance_based/minkowski/mulMul/distance_based/minkowski/ReadVariableOp:value:0&distance_based/minkowski/ones:output:0*
T0*+
_output_shapes
:??????????
3distance_based/minkowski/BroadcastTo/ReadVariableOpReadVariableOp<distance_based_minkowski_broadcastto_readvariableop_resource*
_output_shapes
:*
dtype0?
$distance_based/minkowski/BroadcastToBroadcastTo;distance_based/minkowski/BroadcastTo/ReadVariableOp:value:0'distance_based/minkowski/Shape:output:0*
T0*/
_output_shapes
:?????????
distance_based/minkowski/AbsAbs distance_based/minkowski/sub:z:0*
T0*/
_output_shapes
:?????????r
'distance_based/minkowski/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
#distance_based/minkowski/ExpandDims
ExpandDims distance_based/minkowski/mul:z:00distance_based/minkowski/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
distance_based/minkowski/PowPow distance_based/minkowski/Abs:y:0,distance_based/minkowski/ExpandDims:output:0*
T0*/
_output_shapes
:??????????
distance_based/minkowski/Mul_1Mul distance_based/minkowski/Pow:z:0-distance_based/minkowski/BroadcastTo:output:0*
T0*/
_output_shapes
:?????????y
.distance_based/minkowski/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
distance_based/minkowski/SumSum"distance_based/minkowski/Mul_1:z:07distance_based/minkowski/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(g
"distance_based/minkowski/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
 distance_based/minkowski/truedivRealDiv+distance_based/minkowski/truediv/x:output:0,distance_based/minkowski/ExpandDims:output:0*
T0*/
_output_shapes
:??????????
distance_based/minkowski/Pow_1Pow%distance_based/minkowski/Sum:output:0$distance_based/minkowski/truediv:z:0*
T0*/
_output_shapes
:??????????
!distance_based/minkowski/IdentityIdentity"distance_based/minkowski/Pow_1:z:0*
T0*/
_output_shapes
:??????????
"distance_based/minkowski/IdentityN	IdentityN"distance_based/minkowski/Pow_1:z:0 distance_based/minkowski/sub:z:0-distance_based/minkowski/BroadcastTo:output:0 distance_based/minkowski/mul:z:0*
T
2*,
_gradient_op_typeCustomGradient-117684*|
_output_shapesj
h:?????????:?????????:?????????:??????????
 distance_based/minkowski/SqueezeSqueeze+distance_based/minkowski/IdentityN:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

??????????
8distance_based/exponential_similarity/Neg/ReadVariableOpReadVariableOpAdistance_based_exponential_similarity_neg_readvariableop_resource*
_output_shapes
: *
dtype0?
)distance_based/exponential_similarity/NegNeg@distance_based/exponential_similarity/Neg/ReadVariableOp:value:0*
T0*
_output_shapes
: ?
8distance_based/exponential_similarity/Pow/ReadVariableOpReadVariableOpAdistance_based_exponential_similarity_pow_readvariableop_resource*
_output_shapes
: *
dtype0?
)distance_based/exponential_similarity/PowPow)distance_based/minkowski/Squeeze:output:0@distance_based/exponential_similarity/Pow/ReadVariableOp:value:0*
T0*+
_output_shapes
:??????????
)distance_based/exponential_similarity/mulMul-distance_based/exponential_similarity/Neg:y:0-distance_based/exponential_similarity/Pow:z:0*
T0*+
_output_shapes
:??????????
)distance_based/exponential_similarity/ExpExp-distance_based/exponential_similarity/mul:z:0*
T0*+
_output_shapes
:??????????
8distance_based/exponential_similarity/add/ReadVariableOpReadVariableOpAdistance_based_exponential_similarity_add_readvariableop_resource*
_output_shapes
: *
dtype0?
)distance_based/exponential_similarity/addAddV2-distance_based/exponential_similarity/Exp:y:0@distance_based/exponential_similarity/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????_
CastCastNotEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:??????????
rank_sim_zero_out_nonpresentMul-distance_based/exponential_similarity/add:z:0Cast:y:0*
T0*+
_output_shapes
:??????????

GatherV2_1GatherV2 rank_sim_zero_out_nonpresent:z:0gatherv2_1_indicesgatherv2_axis*
Taxis0*
Tindices0*
Tparams0*/
_output_shapes
:?????????P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :u

ExpandDims
ExpandDimsCast:y:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????G
ConstConst*
_output_shapes
: *
dtype0*
value	B : ?

GatherV2_2GatherV2ExpandDims:output:0Const:output:0gatherv2_axis*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:?????????M
Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B :?
CumsumCumsumGatherV2_1:output:0Cumsum/axis:output:0*
T0*/
_output_shapes
:?????????*
reverse(N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3u
MaximumMaximumGatherV2_1:output:0Maximum/y:output:0*
T0*/
_output_shapes
:?????????P
Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3r
	Maximum_1MaximumCumsum:out:0Maximum_1/y:output:0*
T0*/
_output_shapes
:?????????Q
LogLogMaximum:z:0*
T0*/
_output_shapes
:?????????U
Log_1LogMaximum_1:z:0*
T0*/
_output_shapes
:?????????X
subSubLog:y:0	Log_1:y:0*
T0*/
_output_shapes
:?????????T
mulMulmul_xsub:z:0*
T0*/
_output_shapes
:?????????W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :i
SumSummul:z:0Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????N
ExpExpSum:output:0*
T0*+
_output_shapes
:?????????`
mul_1MulGatherV2_2:output:0Exp:y:0*
T0*+
_output_shapes
:?????????Y
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
Sum_1Sum	mul_1:z:0 Sum_1/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(L
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *    f
EqualEqualSum_1:output:0Equal/y:output:0*
T0*+
_output_shapes
:?????????^
Cast_1Cast	Equal:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????_
truedivRealDiv
Cast_1:y:0	truediv_y*
T0*+
_output_shapes
:?????????Z
addAddV2	mul_1:z:0truediv:z:0*
T0*+
_output_shapes
:?????????`
add_1AddV2Sum_1:output:0
Cast_1:y:0*
T0*+
_output_shapes
:?????????^
	truediv_1RealDivadd:z:0	add_1:z:0*
T0*+
_output_shapes
:?????????`
IdentityIdentitytruediv_1:z:0^NoOp*
T0*+
_output_shapes
:?????????_

Identity_1Identityembedding_variational/mul_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp9^distance_based/exponential_similarity/Neg/ReadVariableOp9^distance_based/exponential_similarity/Pow/ReadVariableOp9^distance_based/exponential_similarity/add/ReadVariableOp4^distance_based/minkowski/BroadcastTo/ReadVariableOp(^distance_based/minkowski/ReadVariableOpu^embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOp~^embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOpq^embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOpz^embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOp=^embedding_variational/embedding_normal_diag/embedding_lookupW^embedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOpP^embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookupj^embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOp+^embedding_variational/mul_1/ReadVariableOp?^embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp?^embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:?????????:?????????:: : : : : : : : : : : : ::: 2t
8distance_based/exponential_similarity/Neg/ReadVariableOp8distance_based/exponential_similarity/Neg/ReadVariableOp2t
8distance_based/exponential_similarity/Pow/ReadVariableOp8distance_based/exponential_similarity/Pow/ReadVariableOp2t
8distance_based/exponential_similarity/add/ReadVariableOp8distance_based/exponential_similarity/add/ReadVariableOp2j
3distance_based/minkowski/BroadcastTo/ReadVariableOp3distance_based/minkowski/BroadcastTo/ReadVariableOp2R
'distance_based/minkowski/ReadVariableOp'distance_based/minkowski/ReadVariableOp2?
tembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOptembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOp2?
}embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOp}embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOp2?
pembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOppembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOp2?
yembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOpyembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOp2|
<embedding_variational/embedding_normal_diag/embedding_lookup<embedding_variational/embedding_normal_diag/embedding_lookup2?
Vembedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOpVembedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOp2?
Oembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookupOembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup2?
iembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOpiembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOp2X
*embedding_variational/mul_1/ReadVariableOp*embedding_variational/mul_1/ReadVariableOp2?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp2?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????
 
_user_specified_nameinputs: 

_output_shapes
::

_output_shapes
: :	

_output_shapes
: :$ 

_output_shapes

::,(
&
_output_shapes
::

_output_shapes
: 
?
?
0__inference_rank_similarity_layer_call_fn_119220
inputs_2rank1_stimulus_set
inputs_agent_id
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5: 
	unknown_6
	unknown_7: 
	unknown_8:
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_2rank1_stimulus_setinputs_agent_idunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:?????????: *,
_read_only_resource_inputs


*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_rank_similarity_layer_call_and_return_conditional_losses_118157s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:?????????:?????????:: : : : : : : : : : : : ::: 22
StatefulPartitionedCallStatefulPartitionedCall:g c
+
_output_shapes
:?????????
4
_user_specified_nameinputs/2rank1/stimulus_set:\X
+
_output_shapes
:?????????
)
_user_specified_nameinputs/agent_id: 

_output_shapes
::

_output_shapes
: :	

_output_shapes
: :$ 

_output_shapes

::,(
&
_output_shapes
::

_output_shapes
: 
??
?
!__inference__wrapped_model_117444
rank1_stimulus_set
agent_id>
:stochastic_behavior_model_rank_similarity_gatherv2_indices;
7stochastic_behavior_model_rank_similarity_gatherv2_axis
mstochastic_behavior_model_rank_similarity_embedding_variational_embedding_normal_diag_embedding_lookup_117160:?
?stochastic_behavior_model_rank_similarity_embedding_variational_embedding_normal_diag_embedding_lookup_1_softplus_readvariableop_resource:?
?stochastic_behavior_model_rank_similarity_embedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_117209:?
?stochastic_behavior_model_rank_similarity_embedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_1_softplus_readvariableop_resource:g
]stochastic_behavior_model_rank_similarity_embedding_variational_mul_1_readvariableop_resource: =
9stochastic_behavior_model_rank_similarity_packed_values_1d
Zstochastic_behavior_model_rank_similarity_distance_based_minkowski_readvariableop_resource: t
fstochastic_behavior_model_rank_similarity_distance_based_minkowski_broadcastto_readvariableop_resource:u
kstochastic_behavior_model_rank_similarity_distance_based_exponential_similarity_neg_readvariableop_resource: u
kstochastic_behavior_model_rank_similarity_distance_based_exponential_similarity_pow_readvariableop_resource: u
kstochastic_behavior_model_rank_similarity_distance_based_exponential_similarity_add_readvariableop_resource: @
<stochastic_behavior_model_rank_similarity_gatherv2_1_indices3
/stochastic_behavior_model_rank_similarity_mul_x7
3stochastic_behavior_model_rank_similarity_truediv_y
identity??bstochastic_behavior_model/rank_similarity/distance_based/exponential_similarity/Neg/ReadVariableOp?bstochastic_behavior_model/rank_similarity/distance_based/exponential_similarity/Pow/ReadVariableOp?bstochastic_behavior_model/rank_similarity/distance_based/exponential_similarity/add/ReadVariableOp?]stochastic_behavior_model/rank_similarity/distance_based/minkowski/BroadcastTo/ReadVariableOp?Qstochastic_behavior_model/rank_similarity/distance_based/minkowski/ReadVariableOp??stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOp??stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOp??stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOp??stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOp?fstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup??stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOp?ystochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup??stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOp?Tstochastic_behavior_model/rank_similarity/embedding_variational/mul_1/ReadVariableOp??stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp??stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp?
2stochastic_behavior_model/rank_similarity/GatherV2GatherV2rank1_stimulus_set:stochastic_behavior_model_rank_similarity_gatherv2_indices7stochastic_behavior_model_rank_similarity_gatherv2_axis*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:?????????v
4stochastic_behavior_model/rank_similarity/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
2stochastic_behavior_model/rank_similarity/NotEqualNotEqual;stochastic_behavior_model/rank_similarity/GatherV2:output:0=stochastic_behavior_model/rank_similarity/NotEqual/y:output:0*
T0*+
_output_shapes
:??????????
fstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookupResourceGathermstochastic_behavior_model_rank_similarity_embedding_variational_embedding_normal_diag_embedding_lookup_117160rank1_stimulus_set*
Tindices0*?
_classv
trloc:@stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup/117160*/
_output_shapes
:?????????*
dtype0?
ostochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup/IdentityIdentityostochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup:output:0*
T0*?
_classv
trloc:@stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup/117160*/
_output_shapes
:??????????
qstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup/Identity_1Identityxstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup/Identity:output:0*
T0*/
_output_shapes
:??????????
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOpReadVariableOp?stochastic_behavior_model_rank_similarity_embedding_variational_embedding_normal_diag_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
qstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/SoftplusSoftplus?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
nstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
lstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/addAddV2wstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/add/x:output:0stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus:activations:0*
T0*
_output_shapes

:?
mstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/axisConst*
_classu
sqloc:@stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/add*
_output_shapes
: *
dtype0*
value	B : ?
hstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1GatherV2pstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/add:z:0rank1_stimulus_setvstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_classu
sqloc:@stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/add*/
_output_shapes
:??????????
qstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/IdentityIdentityqstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1:output:0*
T0*/
_output_shapes
:??????????
pstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
istochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/ShapeShapezstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
istochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
wstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
ystochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
ystochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
qstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_sliceStridedSlicerstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/Shape:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack_1:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
kstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/Shape_1Shapezstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/Identity:output:0*
T0*
_output_shapes
:?
kstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : ?
ystochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
{stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
{stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
sstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1StridedSlicetstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/Shape_1:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack_1:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
tstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
vstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
qstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgsBroadcastArgsstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs/s0_1:output:0zstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice:output:0*
_output_shapes
:?
sstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs_1BroadcastArgsvstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs:r0:0|stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1:output:0*
_output_shapes
:?
sstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
ostochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
jstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/concatConcatV2|stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/concat/values_0:output:0xstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs_1:r0:0xstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
}stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalsstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/concat:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????*
dtype0?
|stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/mulMul?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/RandomStandardNormal:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/stddev:output:0*
T0*N
_output_shapes<
::8?????????????????????????????????????
xstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normalAddV2?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/mul:z:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/mean:output:0*
T0*N
_output_shapes<
::8?????????????????????????????????????
gstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/mulMul|stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal:z:0zstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/Identity:output:0*
T0*E
_output_shapes3
1:/????????????????????????????
gstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/addAddV2kstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/mul:z:0zstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup/Identity_1:output:0*
T0*E
_output_shapes3
1:/????????????????????????????
kstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/Shape_2Shapekstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/add:z:0*
T0*
_output_shapes
:?
ystochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
{stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
{stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
sstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2StridedSlicetstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/Shape_2:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack_1:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
qstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
lstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/concat_1ConcatV2ystochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/sample_shape:output:0|stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2:output:0zstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
kstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/ReshapeReshapekstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/add:z:0ustochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/concat_1:output:0*
T0*/
_output_shapes
:??????????
[stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/zeros_like	ZerosLikerank1_stimulus_set*
T0*+
_output_shapes
:??????????
ystochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookupResourceGather?stochastic_behavior_model_rank_similarity_embedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_117209_stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/zeros_like:y:0*
Tindices0*?
_class?
??loc:@stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/117209*/
_output_shapes
:?????????*
dtype0?
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/IdentityIdentity?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup:output:0*
T0*?
_class?
??loc:@stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/117209*/
_output_shapes
:??????????
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/Identity_1Identity?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/Identity:output:0*
T0*/
_output_shapes
:??????????
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOpReadVariableOp?stochastic_behavior_model_rank_similarity_embedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/SoftplusSoftplus?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/addAddV2?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add/x:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus:activations:0*
T0*
_output_shapes

:?
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/axisConst*?
_class?
??loc:@stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add*
_output_shapes
: *
dtype0*
value	B : ?
{stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1GatherV2?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add:z:0_stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/zeros_like:y:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*?
_class?
??loc:@stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add*/
_output_shapes
:??????????
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/IdentityIdentity?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1:output:0*
T0*/
_output_shapes
:??????????
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
|stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/ShapeShape?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
|stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_sliceStridedSlice?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack_1:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
~stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape_1Shape?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Identity:output:0*
T0*
_output_shapes
:?
~stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : ?
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1StridedSlice?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape_1:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack_1:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgsBroadcastArgs?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs/s0_1:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice:output:0*
_output_shapes
:?
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs_1BroadcastArgs?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs:r0:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1:output:0*
_output_shapes
:?
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
}stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concatConcatV2?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat/values_0:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs_1:r0:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????*
dtype0?
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/mulMul?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/RandomStandardNormal:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/stddev:output:0*
T0*N
_output_shapes<
::8?????????????????????????????????????
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normalAddV2?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/mul:z:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/mean:output:0*
T0*N
_output_shapes<
::8?????????????????????????????????????
zstochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/mulMul?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal:z:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Identity:output:0*
T0*E
_output_shapes3
1:/????????????????????????????
zstochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/addAddV2~stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/mul:z:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/Identity_1:output:0*
T0*E
_output_shapes3
1:/????????????????????????????
~stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape_2Shape~stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/add:z:0*
T0*
_output_shapes
:?
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2StridedSlice?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape_2:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack_1:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat_1ConcatV2?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/sample_shape:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
~stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/ReshapeReshape~stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/add:z:0?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat_1:output:0*
T0*/
_output_shapes
:??????????
Wstochastic_behavior_model/rank_similarity/embedding_variational/reshape/event_shape_outConst*
_output_shapes
: *
dtype0*
valueB ?
Vstochastic_behavior_model/rank_similarity/embedding_variational/reshape/event_shape_inConst*
_output_shapes
:*
dtype0*
valueB"      ?
dstochastic_behavior_model/rank_similarity/embedding_variational/SampleIndependentNormal/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
cstochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/zeroConst*
_output_shapes
: *
dtype0*
value	B : ?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOpReadVariableOpmstochastic_behavior_model_rank_similarity_embedding_variational_embedding_normal_diag_embedding_lookup_117160*
_output_shapes

:*
dtype0?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOpReadVariableOp?stochastic_behavior_model_rank_similarity_embedding_variational_embedding_normal_diag_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/SoftplusSoftplus?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/addAddV2?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add/x:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus:activations:0*
T0*
_output_shapes

:?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"      ?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_sliceStridedSlice?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/shape_as_tensor:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack_1:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"      ?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : ?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1StridedSlice?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/shape_as_tensor_1:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack_1:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgsBroadcastArgs?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs/s0_1:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice:output:0*
_output_shapes
:?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs_1BroadcastArgs?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs:r0:0?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1:output:0*
_output_shapes
:?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concatConcatV2?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat/values_0:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs_1:r0:0?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat:output:0*
T0*"
_output_shapes
:*
dtype0?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/mulMul?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/RandomStandardNormal:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/stddev:output:0*
T0*"
_output_shapes
:?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normalAddV2?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/mul:z:0?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/mean:output:0*
T0*"
_output_shapes
:?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/mulMul?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal:z:0?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add:z:0*
T0*"
_output_shapes
:?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add_1AddV2?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/mul:z:0?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOp:value:0*
T0*"
_output_shapes
:?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReshapeReshape?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add_1:z:0?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Reshape/shape:output:0*
T0*"
_output_shapes
:?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/ReshapeReshape?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Reshape:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Reshape/shape:output:0*
T0*"
_output_shapes
:?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOpReadVariableOp?stochastic_behavior_model_rank_similarity_embedding_variational_embedding_normal_diag_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/SoftplusSoftplus?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/addAddV2?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add/x:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus:activations:0*
T0*
_output_shapes

:?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/truedivRealDiv?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Reshape:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add:z:0*
T0*"
_output_shapes
:?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOpReadVariableOpmstochastic_behavior_model_rank_similarity_embedding_variational_embedding_normal_diag_embedding_lookup_117160*
_output_shapes

:*
dtype0?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/truediv_1RealDiv?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOp:value:0?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add:z:0*
T0*
_output_shapes

:?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/SquaredDifferenceSquaredDifference?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/truediv:z:0?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/truediv_1:z:0*
T0*"
_output_shapes
:?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/mulMul?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/mul/x:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/SquaredDifference:z:0*
T0*"
_output_shapes
:?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??k??
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/LogLog?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add:z:0*
T0*
_output_shapes

:?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add_1AddV2?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Const:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Log:y:0*
T0*
_output_shapes

:?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/subSub?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/mul:z:0?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add_1:z:0*
T0*"
_output_shapes
:?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"?????????
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/SumSum?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/sub:z:0?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Sum/reduction_indices:output:0*
T0*
_output_shapes
:?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/inverse/output_shapeConst*
_output_shapes
:*
dtype0*)
value B"               ?
stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/inverse/ReshapeReshape?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Reshape:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/inverse/output_shape:output:0*
T0**
_output_shapes
:?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/event_shape_tensor/ConstConst*
_output_shapes
:*
dtype0*%
valueB"            ?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/event_shape_tensor/event_shapeIdentity?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/event_shape_tensor/Const:output:0*
T0*
_output_shapes
:?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_event_shape_tensor/IdentityIdentity?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/event_shape_tensor/event_shape:output:0*
T0*
_output_shapes
:?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_event_shape_tensor/output_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"               ?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/ReshapeReshape?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/inverse/Reshape:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Reshape/shape:output:0*
T0**
_output_shapes
:?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                ?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/transpose	Transpose?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Reshape:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/transpose/perm:output:0*
T0**
_output_shapes
:?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOpReadVariableOp?stochastic_behavior_model_rank_similarity_embedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/SoftplusSoftplus?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/addAddV2?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add/x:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus:activations:0*
T0*
_output_shapes

:?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/truedivRealDiv?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/transpose:y:0?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add:z:0*
T0**
_output_shapes
:?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOpReadVariableOp?stochastic_behavior_model_rank_similarity_embedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_117209*
_output_shapes

:*
dtype0?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/truediv_1RealDiv?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp:value:0?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add:z:0*
T0*
_output_shapes

:?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/SquaredDifferenceSquaredDifference?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/truediv:z:0?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/truediv_1:z:0*
T0**
_output_shapes
:?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/mulMul?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/mul/x:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/SquaredDifference:z:0*
T0**
_output_shapes
:?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??k??
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/LogLog?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add:z:0*
T0*
_output_shapes

:?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add_1AddV2?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Const:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Log:y:0*
T0*
_output_shapes

:?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/subSub?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/mul:z:0?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add_1:z:0*
T0**
_output_shapes
:?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"?????????
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/SumSum?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/sub:z:0?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Sum/reduction_indices:output:0*
T0*"
_output_shapes
:?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/BroadcastTo/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/BroadcastToBroadcastTo?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Sum:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/BroadcastTo/shape:output:0*
T0*"
_output_shapes
:?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/SumSum?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/BroadcastTo:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Sum/reduction_indices:output:0*
T0*
_output_shapes
:?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/NegNeg?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/zeros:output:0*
T0*
_output_shapes
: ?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Neg_1Neg?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Neg:y:0*
T0*
_output_shapes
: ?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/onesConst*
_output_shapes

:*
dtype0*
valueB*  ???
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/mulMul?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/ones:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Neg_1:y:0*
T0*
_output_shapes

:?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"?????????
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/SumSum?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/mul:z:0?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Sum/reduction_indices:output:0*
T0*
_output_shapes
: ?
kstochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/subSub?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Sum:output:0?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Sum:output:0*
T0*
_output_shapes
:?
Cstochastic_behavior_model/rank_similarity/embedding_variational/subSub?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Sum:output:0ostochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/sub:z:0*
T0*
_output_shapes
:?
Estochastic_behavior_model/rank_similarity/embedding_variational/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Dstochastic_behavior_model/rank_similarity/embedding_variational/MeanMeanGstochastic_behavior_model/rank_similarity/embedding_variational/sub:z:0Nstochastic_behavior_model/rank_similarity/embedding_variational/Const:output:0*
T0*
_output_shapes
: ?
Estochastic_behavior_model/rank_similarity/embedding_variational/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *x?V9?
Cstochastic_behavior_model/rank_similarity/embedding_variational/mulMulMstochastic_behavior_model/rank_similarity/embedding_variational/Mean:output:0Nstochastic_behavior_model/rank_similarity/embedding_variational/mul/y:output:0*
T0*
_output_shapes
: ?
Tstochastic_behavior_model/rank_similarity/embedding_variational/mul_1/ReadVariableOpReadVariableOp]stochastic_behavior_model_rank_similarity_embedding_variational_mul_1_readvariableop_resource*
_output_shapes
: *
dtype0?
Estochastic_behavior_model/rank_similarity/embedding_variational/mul_1MulGstochastic_behavior_model/rank_similarity/embedding_variational/mul:z:0\stochastic_behavior_model/rank_similarity/embedding_variational/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
: t
2stochastic_behavior_model/rank_similarity/packed/0Const*
_output_shapes
: *
dtype0*
value	B :?
0stochastic_behavior_model/rank_similarity/packedPack;stochastic_behavior_model/rank_similarity/packed/0:output:09stochastic_behavior_model_rank_similarity_packed_values_1*
N*
T0*
_output_shapes
:{
9stochastic_behavior_model/rank_similarity/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
/stochastic_behavior_model/rank_similarity/splitSplitVtstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/Reshape:output:09stochastic_behavior_model/rank_similarity/packed:output:0Bstochastic_behavior_model/rank_similarity/split/split_dim:output:0*
T0*

Tlen0*J
_output_shapes8
6:?????????:?????????*
	num_split?
Fstochastic_behavior_model/rank_similarity/distance_based/minkowski/subSub8stochastic_behavior_model/rank_similarity/split:output:08stochastic_behavior_model/rank_similarity/split:output:1*
T0*/
_output_shapes
:??????????
Hstochastic_behavior_model/rank_similarity/distance_based/minkowski/ShapeShapeJstochastic_behavior_model/rank_similarity/distance_based/minkowski/sub:z:0*
T0*
_output_shapes
:?
Vstochastic_behavior_model/rank_similarity/distance_based/minkowski/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Xstochastic_behavior_model/rank_similarity/distance_based/minkowski/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
Xstochastic_behavior_model/rank_similarity/distance_based/minkowski/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Pstochastic_behavior_model/rank_similarity/distance_based/minkowski/strided_sliceStridedSliceQstochastic_behavior_model/rank_similarity/distance_based/minkowski/Shape:output:0_stochastic_behavior_model/rank_similarity/distance_based/minkowski/strided_slice/stack:output:0astochastic_behavior_model/rank_similarity/distance_based/minkowski/strided_slice/stack_1:output:0astochastic_behavior_model/rank_similarity/distance_based/minkowski/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:?
Mstochastic_behavior_model/rank_similarity/distance_based/minkowski/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Gstochastic_behavior_model/rank_similarity/distance_based/minkowski/onesFillYstochastic_behavior_model/rank_similarity/distance_based/minkowski/strided_slice:output:0Vstochastic_behavior_model/rank_similarity/distance_based/minkowski/ones/Const:output:0*
T0*+
_output_shapes
:??????????
Qstochastic_behavior_model/rank_similarity/distance_based/minkowski/ReadVariableOpReadVariableOpZstochastic_behavior_model_rank_similarity_distance_based_minkowski_readvariableop_resource*
_output_shapes
: *
dtype0?
Fstochastic_behavior_model/rank_similarity/distance_based/minkowski/mulMulYstochastic_behavior_model/rank_similarity/distance_based/minkowski/ReadVariableOp:value:0Pstochastic_behavior_model/rank_similarity/distance_based/minkowski/ones:output:0*
T0*+
_output_shapes
:??????????
]stochastic_behavior_model/rank_similarity/distance_based/minkowski/BroadcastTo/ReadVariableOpReadVariableOpfstochastic_behavior_model_rank_similarity_distance_based_minkowski_broadcastto_readvariableop_resource*
_output_shapes
:*
dtype0?
Nstochastic_behavior_model/rank_similarity/distance_based/minkowski/BroadcastToBroadcastToestochastic_behavior_model/rank_similarity/distance_based/minkowski/BroadcastTo/ReadVariableOp:value:0Qstochastic_behavior_model/rank_similarity/distance_based/minkowski/Shape:output:0*
T0*/
_output_shapes
:??????????
Fstochastic_behavior_model/rank_similarity/distance_based/minkowski/AbsAbsJstochastic_behavior_model/rank_similarity/distance_based/minkowski/sub:z:0*
T0*/
_output_shapes
:??????????
Qstochastic_behavior_model/rank_similarity/distance_based/minkowski/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Mstochastic_behavior_model/rank_similarity/distance_based/minkowski/ExpandDims
ExpandDimsJstochastic_behavior_model/rank_similarity/distance_based/minkowski/mul:z:0Zstochastic_behavior_model/rank_similarity/distance_based/minkowski/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
Fstochastic_behavior_model/rank_similarity/distance_based/minkowski/PowPowJstochastic_behavior_model/rank_similarity/distance_based/minkowski/Abs:y:0Vstochastic_behavior_model/rank_similarity/distance_based/minkowski/ExpandDims:output:0*
T0*/
_output_shapes
:??????????
Hstochastic_behavior_model/rank_similarity/distance_based/minkowski/Mul_1MulJstochastic_behavior_model/rank_similarity/distance_based/minkowski/Pow:z:0Wstochastic_behavior_model/rank_similarity/distance_based/minkowski/BroadcastTo:output:0*
T0*/
_output_shapes
:??????????
Xstochastic_behavior_model/rank_similarity/distance_based/minkowski/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Fstochastic_behavior_model/rank_similarity/distance_based/minkowski/SumSumLstochastic_behavior_model/rank_similarity/distance_based/minkowski/Mul_1:z:0astochastic_behavior_model/rank_similarity/distance_based/minkowski/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(?
Lstochastic_behavior_model/rank_similarity/distance_based/minkowski/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Jstochastic_behavior_model/rank_similarity/distance_based/minkowski/truedivRealDivUstochastic_behavior_model/rank_similarity/distance_based/minkowski/truediv/x:output:0Vstochastic_behavior_model/rank_similarity/distance_based/minkowski/ExpandDims:output:0*
T0*/
_output_shapes
:??????????
Hstochastic_behavior_model/rank_similarity/distance_based/minkowski/Pow_1PowOstochastic_behavior_model/rank_similarity/distance_based/minkowski/Sum:output:0Nstochastic_behavior_model/rank_similarity/distance_based/minkowski/truediv:z:0*
T0*/
_output_shapes
:??????????
Kstochastic_behavior_model/rank_similarity/distance_based/minkowski/IdentityIdentityLstochastic_behavior_model/rank_similarity/distance_based/minkowski/Pow_1:z:0*
T0*/
_output_shapes
:??????????
Lstochastic_behavior_model/rank_similarity/distance_based/minkowski/IdentityN	IdentityNLstochastic_behavior_model/rank_similarity/distance_based/minkowski/Pow_1:z:0Jstochastic_behavior_model/rank_similarity/distance_based/minkowski/sub:z:0Wstochastic_behavior_model/rank_similarity/distance_based/minkowski/BroadcastTo:output:0Jstochastic_behavior_model/rank_similarity/distance_based/minkowski/mul:z:0*
T
2*,
_gradient_op_typeCustomGradient-117382*|
_output_shapesj
h:?????????:?????????:?????????:??????????
Jstochastic_behavior_model/rank_similarity/distance_based/minkowski/SqueezeSqueezeUstochastic_behavior_model/rank_similarity/distance_based/minkowski/IdentityN:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

??????????
bstochastic_behavior_model/rank_similarity/distance_based/exponential_similarity/Neg/ReadVariableOpReadVariableOpkstochastic_behavior_model_rank_similarity_distance_based_exponential_similarity_neg_readvariableop_resource*
_output_shapes
: *
dtype0?
Sstochastic_behavior_model/rank_similarity/distance_based/exponential_similarity/NegNegjstochastic_behavior_model/rank_similarity/distance_based/exponential_similarity/Neg/ReadVariableOp:value:0*
T0*
_output_shapes
: ?
bstochastic_behavior_model/rank_similarity/distance_based/exponential_similarity/Pow/ReadVariableOpReadVariableOpkstochastic_behavior_model_rank_similarity_distance_based_exponential_similarity_pow_readvariableop_resource*
_output_shapes
: *
dtype0?
Sstochastic_behavior_model/rank_similarity/distance_based/exponential_similarity/PowPowSstochastic_behavior_model/rank_similarity/distance_based/minkowski/Squeeze:output:0jstochastic_behavior_model/rank_similarity/distance_based/exponential_similarity/Pow/ReadVariableOp:value:0*
T0*+
_output_shapes
:??????????
Sstochastic_behavior_model/rank_similarity/distance_based/exponential_similarity/mulMulWstochastic_behavior_model/rank_similarity/distance_based/exponential_similarity/Neg:y:0Wstochastic_behavior_model/rank_similarity/distance_based/exponential_similarity/Pow:z:0*
T0*+
_output_shapes
:??????????
Sstochastic_behavior_model/rank_similarity/distance_based/exponential_similarity/ExpExpWstochastic_behavior_model/rank_similarity/distance_based/exponential_similarity/mul:z:0*
T0*+
_output_shapes
:??????????
bstochastic_behavior_model/rank_similarity/distance_based/exponential_similarity/add/ReadVariableOpReadVariableOpkstochastic_behavior_model_rank_similarity_distance_based_exponential_similarity_add_readvariableop_resource*
_output_shapes
: *
dtype0?
Sstochastic_behavior_model/rank_similarity/distance_based/exponential_similarity/addAddV2Wstochastic_behavior_model/rank_similarity/distance_based/exponential_similarity/Exp:y:0jstochastic_behavior_model/rank_similarity/distance_based/exponential_similarity/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:??????????
.stochastic_behavior_model/rank_similarity/CastCast6stochastic_behavior_model/rank_similarity/NotEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:??????????
Fstochastic_behavior_model/rank_similarity/rank_sim_zero_out_nonpresentMulWstochastic_behavior_model/rank_similarity/distance_based/exponential_similarity/add:z:02stochastic_behavior_model/rank_similarity/Cast:y:0*
T0*+
_output_shapes
:??????????
4stochastic_behavior_model/rank_similarity/GatherV2_1GatherV2Jstochastic_behavior_model/rank_similarity/rank_sim_zero_out_nonpresent:z:0<stochastic_behavior_model_rank_similarity_gatherv2_1_indices7stochastic_behavior_model_rank_similarity_gatherv2_axis*
Taxis0*
Tindices0*
Tparams0*/
_output_shapes
:?????????z
8stochastic_behavior_model/rank_similarity/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
4stochastic_behavior_model/rank_similarity/ExpandDims
ExpandDims2stochastic_behavior_model/rank_similarity/Cast:y:0Astochastic_behavior_model/rank_similarity/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????q
/stochastic_behavior_model/rank_similarity/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
4stochastic_behavior_model/rank_similarity/GatherV2_2GatherV2=stochastic_behavior_model/rank_similarity/ExpandDims:output:08stochastic_behavior_model/rank_similarity/Const:output:07stochastic_behavior_model_rank_similarity_gatherv2_axis*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:?????????w
5stochastic_behavior_model/rank_similarity/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B :?
0stochastic_behavior_model/rank_similarity/CumsumCumsum=stochastic_behavior_model/rank_similarity/GatherV2_1:output:0>stochastic_behavior_model/rank_similarity/Cumsum/axis:output:0*
T0*/
_output_shapes
:?????????*
reverse(x
3stochastic_behavior_model/rank_similarity/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
1stochastic_behavior_model/rank_similarity/MaximumMaximum=stochastic_behavior_model/rank_similarity/GatherV2_1:output:0<stochastic_behavior_model/rank_similarity/Maximum/y:output:0*
T0*/
_output_shapes
:?????????z
5stochastic_behavior_model/rank_similarity/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
3stochastic_behavior_model/rank_similarity/Maximum_1Maximum6stochastic_behavior_model/rank_similarity/Cumsum:out:0>stochastic_behavior_model/rank_similarity/Maximum_1/y:output:0*
T0*/
_output_shapes
:??????????
-stochastic_behavior_model/rank_similarity/LogLog5stochastic_behavior_model/rank_similarity/Maximum:z:0*
T0*/
_output_shapes
:??????????
/stochastic_behavior_model/rank_similarity/Log_1Log7stochastic_behavior_model/rank_similarity/Maximum_1:z:0*
T0*/
_output_shapes
:??????????
-stochastic_behavior_model/rank_similarity/subSub1stochastic_behavior_model/rank_similarity/Log:y:03stochastic_behavior_model/rank_similarity/Log_1:y:0*
T0*/
_output_shapes
:??????????
-stochastic_behavior_model/rank_similarity/mulMul/stochastic_behavior_model_rank_similarity_mul_x1stochastic_behavior_model/rank_similarity/sub:z:0*
T0*/
_output_shapes
:??????????
?stochastic_behavior_model/rank_similarity/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
-stochastic_behavior_model/rank_similarity/SumSum1stochastic_behavior_model/rank_similarity/mul:z:0Hstochastic_behavior_model/rank_similarity/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:??????????
-stochastic_behavior_model/rank_similarity/ExpExp6stochastic_behavior_model/rank_similarity/Sum:output:0*
T0*+
_output_shapes
:??????????
/stochastic_behavior_model/rank_similarity/mul_1Mul=stochastic_behavior_model/rank_similarity/GatherV2_2:output:01stochastic_behavior_model/rank_similarity/Exp:y:0*
T0*+
_output_shapes
:??????????
Astochastic_behavior_model/rank_similarity/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
/stochastic_behavior_model/rank_similarity/Sum_1Sum3stochastic_behavior_model/rank_similarity/mul_1:z:0Jstochastic_behavior_model/rank_similarity/Sum_1/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(v
1stochastic_behavior_model/rank_similarity/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
/stochastic_behavior_model/rank_similarity/EqualEqual8stochastic_behavior_model/rank_similarity/Sum_1:output:0:stochastic_behavior_model/rank_similarity/Equal/y:output:0*
T0*+
_output_shapes
:??????????
0stochastic_behavior_model/rank_similarity/Cast_1Cast3stochastic_behavior_model/rank_similarity/Equal:z:0*

DstT0*

SrcT0
*+
_output_shapes
:??????????
1stochastic_behavior_model/rank_similarity/truedivRealDiv4stochastic_behavior_model/rank_similarity/Cast_1:y:03stochastic_behavior_model_rank_similarity_truediv_y*
T0*+
_output_shapes
:??????????
-stochastic_behavior_model/rank_similarity/addAddV23stochastic_behavior_model/rank_similarity/mul_1:z:05stochastic_behavior_model/rank_similarity/truediv:z:0*
T0*+
_output_shapes
:??????????
/stochastic_behavior_model/rank_similarity/add_1AddV28stochastic_behavior_model/rank_similarity/Sum_1:output:04stochastic_behavior_model/rank_similarity/Cast_1:y:0*
T0*+
_output_shapes
:??????????
3stochastic_behavior_model/rank_similarity/truediv_1RealDiv1stochastic_behavior_model/rank_similarity/add:z:03stochastic_behavior_model/rank_similarity/add_1:z:0*
T0*+
_output_shapes
:??????????
IdentityIdentity7stochastic_behavior_model/rank_similarity/truediv_1:z:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOpc^stochastic_behavior_model/rank_similarity/distance_based/exponential_similarity/Neg/ReadVariableOpc^stochastic_behavior_model/rank_similarity/distance_based/exponential_similarity/Pow/ReadVariableOpc^stochastic_behavior_model/rank_similarity/distance_based/exponential_similarity/add/ReadVariableOp^^stochastic_behavior_model/rank_similarity/distance_based/minkowski/BroadcastTo/ReadVariableOpR^stochastic_behavior_model/rank_similarity/distance_based/minkowski/ReadVariableOp?^stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOp?^stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOp?^stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOp?^stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOpg^stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup?^stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOpz^stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup?^stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOpU^stochastic_behavior_model/rank_similarity/embedding_variational/mul_1/ReadVariableOp?^stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp?^stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:?????????:?????????:: : : : : : : : : : : : ::: 2?
bstochastic_behavior_model/rank_similarity/distance_based/exponential_similarity/Neg/ReadVariableOpbstochastic_behavior_model/rank_similarity/distance_based/exponential_similarity/Neg/ReadVariableOp2?
bstochastic_behavior_model/rank_similarity/distance_based/exponential_similarity/Pow/ReadVariableOpbstochastic_behavior_model/rank_similarity/distance_based/exponential_similarity/Pow/ReadVariableOp2?
bstochastic_behavior_model/rank_similarity/distance_based/exponential_similarity/add/ReadVariableOpbstochastic_behavior_model/rank_similarity/distance_based/exponential_similarity/add/ReadVariableOp2?
]stochastic_behavior_model/rank_similarity/distance_based/minkowski/BroadcastTo/ReadVariableOp]stochastic_behavior_model/rank_similarity/distance_based/minkowski/BroadcastTo/ReadVariableOp2?
Qstochastic_behavior_model/rank_similarity/distance_based/minkowski/ReadVariableOpQstochastic_behavior_model/rank_similarity/distance_based/minkowski/ReadVariableOp2?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOp?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOp2?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOp?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOp2?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOp?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOp2?
?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOp?stochastic_behavior_model/rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOp2?
fstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookupfstochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup2?
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOp?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOp2?
ystochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookupystochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup2?
?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOp?stochastic_behavior_model/rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOp2?
Tstochastic_behavior_model/rank_similarity/embedding_variational/mul_1/ReadVariableOpTstochastic_behavior_model/rank_similarity/embedding_variational/mul_1/ReadVariableOp2?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp2?
?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp?stochastic_behavior_model/rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp:` \
+
_output_shapes
:?????????
-
_user_specified_name2rank1/stimulus_set:UQ
+
_output_shapes
:?????????
"
_user_specified_name
agent_id: 

_output_shapes
::

_output_shapes
: :	

_output_shapes
: :$ 

_output_shapes

::,(
&
_output_shapes
::

_output_shapes
: 
??
?
K__inference_rank_similarity_layer_call_and_return_conditional_losses_119514
inputs_2rank1_stimulus_set
inputs_agent_id
gatherv2_indices
gatherv2_axisU
Cembedding_variational_embedding_normal_diag_embedding_lookup_119229:q
_embedding_variational_embedding_normal_diag_embedding_lookup_1_softplus_readvariableop_resource:h
Vembedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_119278:?
rembedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_1_softplus_readvariableop_resource:=
3embedding_variational_mul_1_readvariableop_resource: 
packed_values_1:
0distance_based_minkowski_readvariableop_resource: J
<distance_based_minkowski_broadcastto_readvariableop_resource:K
Adistance_based_exponential_similarity_neg_readvariableop_resource: K
Adistance_based_exponential_similarity_pow_readvariableop_resource: K
Adistance_based_exponential_similarity_add_readvariableop_resource: 
gatherv2_1_indices	
mul_x
	truediv_y
identity

identity_1??8distance_based/exponential_similarity/Neg/ReadVariableOp?8distance_based/exponential_similarity/Pow/ReadVariableOp?8distance_based/exponential_similarity/add/ReadVariableOp?3distance_based/minkowski/BroadcastTo/ReadVariableOp?'distance_based/minkowski/ReadVariableOp?tembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOp?}embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOp?pembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOp?yembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOp?<embedding_variational/embedding_normal_diag/embedding_lookup?Vembedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOp?Oembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup?iembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOp?*embedding_variational/mul_1/ReadVariableOp??embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp??embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp?
GatherV2GatherV2inputs_2rank1_stimulus_setgatherv2_indicesgatherv2_axis*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:?????????L

NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : r
NotEqualNotEqualGatherV2:output:0NotEqual/y:output:0*
T0*+
_output_shapes
:??????????
<embedding_variational/embedding_normal_diag/embedding_lookupResourceGatherCembedding_variational_embedding_normal_diag_embedding_lookup_119229inputs_2rank1_stimulus_set*
Tindices0*V
_classL
JHloc:@embedding_variational/embedding_normal_diag/embedding_lookup/119229*/
_output_shapes
:?????????*
dtype0?
Eembedding_variational/embedding_normal_diag/embedding_lookup/IdentityIdentityEembedding_variational/embedding_normal_diag/embedding_lookup:output:0*
T0*V
_classL
JHloc:@embedding_variational/embedding_normal_diag/embedding_lookup/119229*/
_output_shapes
:??????????
Gembedding_variational/embedding_normal_diag/embedding_lookup/Identity_1IdentityNembedding_variational/embedding_normal_diag/embedding_lookup/Identity:output:0*
T0*/
_output_shapes
:??????????
Vembedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOpReadVariableOp_embedding_variational_embedding_normal_diag_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
Gembedding_variational/embedding_normal_diag/embedding_lookup_1/SoftplusSoftplus^embedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
Dembedding_variational/embedding_normal_diag/embedding_lookup_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
Bembedding_variational/embedding_normal_diag/embedding_lookup_1/addAddV2Membedding_variational/embedding_normal_diag/embedding_lookup_1/add/x:output:0Uembedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus:activations:0*
T0*
_output_shapes

:?
Cembedding_variational/embedding_normal_diag/embedding_lookup_1/axisConst*U
_classK
IGloc:@embedding_variational/embedding_normal_diag/embedding_lookup_1/add*
_output_shapes
: *
dtype0*
value	B : ?
>embedding_variational/embedding_normal_diag/embedding_lookup_1GatherV2Fembedding_variational/embedding_normal_diag/embedding_lookup_1/add:z:0inputs_2rank1_stimulus_setLembedding_variational/embedding_normal_diag/embedding_lookup_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*U
_classK
IGloc:@embedding_variational/embedding_normal_diag/embedding_lookup_1/add*/
_output_shapes
:??????????
Gembedding_variational/embedding_normal_diag/embedding_lookup_1/IdentityIdentityGembedding_variational/embedding_normal_diag/embedding_lookup_1:output:0*
T0*/
_output_shapes
:??????????
Fembedding_variational/embedding_normal_diag/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
?embedding_variational/embedding_normal_diag/Normal/sample/ShapeShapePembedding_variational/embedding_normal_diag/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
?embedding_variational/embedding_normal_diag/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
Membedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Oembedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Oembedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Gembedding_variational/embedding_normal_diag/Normal/sample/strided_sliceStridedSliceHembedding_variational/embedding_normal_diag/Normal/sample/Shape:output:0Vembedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack:output:0Xembedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack_1:output:0Xembedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Aembedding_variational/embedding_normal_diag/Normal/sample/Shape_1ShapePembedding_variational/embedding_normal_diag/embedding_lookup_1/Identity:output:0*
T0*
_output_shapes
:?
Aembedding_variational/embedding_normal_diag/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : ?
Oembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Qembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Qembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Iembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1StridedSliceJembedding_variational/embedding_normal_diag/Normal/sample/Shape_1:output:0Xembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack:output:0Zembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack_1:output:0Zembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Jembedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
Lembedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
Gembedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgsBroadcastArgsUembedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs/s0_1:output:0Pembedding_variational/embedding_normal_diag/Normal/sample/strided_slice:output:0*
_output_shapes
:?
Iembedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs_1BroadcastArgsLembedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs:r0:0Rembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1:output:0*
_output_shapes
:?
Iembedding_variational/embedding_normal_diag/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
Eembedding_variational/embedding_normal_diag/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
@embedding_variational/embedding_normal_diag/Normal/sample/concatConcatV2Rembedding_variational/embedding_normal_diag/Normal/sample/concat/values_0:output:0Nembedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs_1:r0:0Nembedding_variational/embedding_normal_diag/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Sembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
Uembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
cembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalIembedding_variational/embedding_normal_diag/Normal/sample/concat:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????*
dtype0?
Rembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/mulMullembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/RandomStandardNormal:output:0^embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/stddev:output:0*
T0*N
_output_shapes<
::8?????????????????????????????????????
Nembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normalAddV2Vembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/mul:z:0\embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/mean:output:0*
T0*N
_output_shapes<
::8?????????????????????????????????????
=embedding_variational/embedding_normal_diag/Normal/sample/mulMulRembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal:z:0Pembedding_variational/embedding_normal_diag/embedding_lookup_1/Identity:output:0*
T0*E
_output_shapes3
1:/????????????????????????????
=embedding_variational/embedding_normal_diag/Normal/sample/addAddV2Aembedding_variational/embedding_normal_diag/Normal/sample/mul:z:0Pembedding_variational/embedding_normal_diag/embedding_lookup/Identity_1:output:0*
T0*E
_output_shapes3
1:/????????????????????????????
Aembedding_variational/embedding_normal_diag/Normal/sample/Shape_2ShapeAembedding_variational/embedding_normal_diag/Normal/sample/add:z:0*
T0*
_output_shapes
:?
Oembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Qembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Qembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Iembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2StridedSliceJembedding_variational/embedding_normal_diag/Normal/sample/Shape_2:output:0Xembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack:output:0Zembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack_1:output:0Zembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Gembedding_variational/embedding_normal_diag/Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Bembedding_variational/embedding_normal_diag/Normal/sample/concat_1ConcatV2Oembedding_variational/embedding_normal_diag/Normal/sample/sample_shape:output:0Rembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2:output:0Pembedding_variational/embedding_normal_diag/Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Aembedding_variational/embedding_normal_diag/Normal/sample/ReshapeReshapeAembedding_variational/embedding_normal_diag/Normal/sample/add:z:0Kembedding_variational/embedding_normal_diag/Normal/sample/concat_1:output:0*
T0*/
_output_shapes
:??????????
1embedding_variational/embedding_shared/zeros_like	ZerosLikeinputs_2rank1_stimulus_set*
T0*+
_output_shapes
:??????????
Oembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookupResourceGatherVembedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_1192785embedding_variational/embedding_shared/zeros_like:y:0*
Tindices0*i
_class_
][loc:@embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/119278*/
_output_shapes
:?????????*
dtype0?
Xembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/IdentityIdentityXembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup:output:0*
T0*i
_class_
][loc:@embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/119278*/
_output_shapes
:??????????
Zembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/Identity_1Identityaembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/Identity:output:0*
T0*/
_output_shapes
:??????????
iembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOpReadVariableOprembedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
Zembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/SoftplusSoftplusqembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
Wembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
Uembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/addAddV2`embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add/x:output:0hembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus:activations:0*
T0*
_output_shapes

:?
Vembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/axisConst*h
_class^
\Zloc:@embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add*
_output_shapes
: *
dtype0*
value	B : ?
Qembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1GatherV2Yembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add:z:05embedding_variational/embedding_shared/zeros_like:y:0_embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*h
_class^
\Zloc:@embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add*/
_output_shapes
:??????????
Zembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/IdentityIdentityZembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1:output:0*
T0*/
_output_shapes
:??????????
Yembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Rembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/ShapeShapecembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
Rembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
`embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
bembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
bembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Zembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_sliceStridedSlice[embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape:output:0iembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack:output:0kembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack_1:output:0kembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Tembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape_1Shapecembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Identity:output:0*
T0*
_output_shapes
:?
Tembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : ?
bembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
dembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
dembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1StridedSlice]embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape_1:output:0kembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack:output:0membedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack_1:output:0membedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
]embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
_embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
Zembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgsBroadcastArgshembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs/s0_1:output:0cembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice:output:0*
_output_shapes
:?
\embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs_1BroadcastArgs_embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs:r0:0eembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1:output:0*
_output_shapes
:?
\embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
Xembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Sembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concatConcatV2eembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat/values_0:output:0aembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs_1:r0:0aembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
fembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
hembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
vembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal\embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????*
dtype0?
eembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/mulMulembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/RandomStandardNormal:output:0qembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/stddev:output:0*
T0*N
_output_shapes<
::8?????????????????????????????????????
aembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normalAddV2iembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/mul:z:0oembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/mean:output:0*
T0*N
_output_shapes<
::8?????????????????????????????????????
Pembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/mulMuleembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal:z:0cembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Identity:output:0*
T0*E
_output_shapes3
1:/????????????????????????????
Pembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/addAddV2Tembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/mul:z:0cembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/Identity_1:output:0*
T0*E
_output_shapes3
1:/????????????????????????????
Tembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape_2ShapeTembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/add:z:0*
T0*
_output_shapes
:?
bembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
dembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2StridedSlice]embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape_2:output:0kembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack:output:0membedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack_1:output:0membedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Zembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Uembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat_1ConcatV2bembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/sample_shape:output:0eembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2:output:0cembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Tembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/ReshapeReshapeTembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/add:z:0^embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat_1:output:0*
T0*/
_output_shapes
:?????????p
-embedding_variational/reshape/event_shape_outConst*
_output_shapes
: *
dtype0*
valueB }
,embedding_variational/reshape/event_shape_inConst*
_output_shapes
:*
dtype0*
valueB"      ?
:embedding_variational/SampleIndependentNormal/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB"      {
9embedding_variational/reshapeSampleIndependentNormal/zeroConst*
_output_shapes
: *
dtype0*
value	B : ?
`embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
qembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
nembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
pembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOpReadVariableOpCembedding_variational_embedding_normal_diag_embedding_lookup_119229*
_output_shapes

:*
dtype0?
yembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOpReadVariableOp_embedding_variational_embedding_normal_diag_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
jembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/SoftplusSoftplus?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
gembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
eembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/addAddV2pembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add/x:output:0xembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus:activations:0*
T0*
_output_shapes

:?
qembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"      ?
gembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
uembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
wembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
wembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
oembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_sliceStridedSlicezembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/shape_as_tensor:output:0~embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack:output:0?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack_1:output:0?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
sembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"      ?
iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : ?
wembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
yembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
yembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
qembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1StridedSlice|embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/shape_as_tensor_1:output:0?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack:output:0?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack_1:output:0?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
rembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
tembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
oembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgsBroadcastArgs}embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs/s0_1:output:0xembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice:output:0*
_output_shapes
:?
qembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs_1BroadcastArgstembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs:r0:0zembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1:output:0*
_output_shapes
:?
qembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
membedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
hembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concatConcatV2zembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat/values_0:output:0vembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs_1:r0:0vembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
{embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
}embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalqembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat:output:0*
T0*"
_output_shapes
:*
dtype0?
zembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/mulMul?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/RandomStandardNormal:output:0?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/stddev:output:0*
T0*"
_output_shapes
:?
vembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normalAddV2~embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/mul:z:0?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/mean:output:0*
T0*"
_output_shapes
:?
eembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/mulMulzembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal:z:0iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add:z:0*
T0*"
_output_shapes
:?
gembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add_1AddV2iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/mul:z:0xembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOp:value:0*
T0*"
_output_shapes
:?
oembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReshapeReshapekembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add_1:z:0xembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Reshape/shape:output:0*
T0*"
_output_shapes
:?
aembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
[embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/ReshapeReshaperembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Reshape:output:0jembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Reshape/shape:output:0*
T0*"
_output_shapes
:?
}embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOpReadVariableOp_embedding_variational_embedding_normal_diag_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
nembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/SoftplusSoftplus?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
kembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/addAddV2tembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add/x:output:0|embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus:activations:0*
T0*
_output_shapes

:?
membedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/truedivRealDivdembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Reshape:output:0membedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add:z:0*
T0*"
_output_shapes
:?
tembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOpReadVariableOpCembedding_variational_embedding_normal_diag_embedding_lookup_119229*
_output_shapes

:*
dtype0?
oembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/truediv_1RealDiv|embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOp:value:0membedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add:z:0*
T0*
_output_shapes

:?
wembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/SquaredDifferenceSquaredDifferenceqembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/truediv:z:0sembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/truediv_1:z:0*
T0*"
_output_shapes
:?
kembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/mulMultembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/mul/x:output:0{embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/SquaredDifference:z:0*
T0*"
_output_shapes
:?
kembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??k??
iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/LogLogmembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add:z:0*
T0*
_output_shapes

:?
kembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add_1AddV2tembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Const:output:0membedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Log:y:0*
T0*
_output_shapes

:?
iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/subSubmembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/mul:z:0oembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add_1:z:0*
T0*"
_output_shapes
:?
kembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"?????????
Yembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/SumSummembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/sub:z:0tembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Sum/reduction_indices:output:0*
T0*
_output_shapes
:?
Zembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/inverse/output_shapeConst*
_output_shapes
:*
dtype0*)
value B"               ?
Uembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/inverse/ReshapeReshapedembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Reshape:output:0cembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/inverse/output_shape:output:0*
T0**
_output_shapes
:?
nembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/event_shape_tensor/ConstConst*
_output_shapes
:*
dtype0*%
valueB"            ?
tembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/event_shape_tensor/event_shapeIdentitywembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/event_shape_tensor/Const:output:0*
T0*
_output_shapes
:?
iembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_event_shape_tensor/IdentityIdentity}embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/event_shape_tensor/event_shape:output:0*
T0*
_output_shapes
:?
membedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_event_shape_tensor/output_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
lembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"               ?
fembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/ReshapeReshape^embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/inverse/Reshape:output:0uembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Reshape/shape:output:0*
T0**
_output_shapes
:?
membedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                ?
hembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/transpose	Transposeoembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Reshape:output:0vembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/transpose/perm:output:0*
T0**
_output_shapes
:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOpReadVariableOprembedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/SoftplusSoftplus?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/addAddV2?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add/x:output:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus:activations:0*
T0*
_output_shapes

:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/truedivRealDivlembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/transpose:y:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add:z:0*
T0**
_output_shapes
:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOpReadVariableOpVembedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_119278*
_output_shapes

:*
dtype0?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/truediv_1RealDiv?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp:value:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add:z:0*
T0*
_output_shapes

:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/SquaredDifferenceSquaredDifference?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/truediv:z:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/truediv_1:z:0*
T0**
_output_shapes
:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/mulMul?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/mul/x:output:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/SquaredDifference:z:0*
T0**
_output_shapes
:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??k??
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/LogLog?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add:z:0*
T0*
_output_shapes

:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add_1AddV2?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Const:output:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Log:y:0*
T0*
_output_shapes

:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/subSub?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/mul:z:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add_1:z:0*
T0**
_output_shapes
:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"?????????
}embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/SumSum?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/sub:z:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Sum/reduction_indices:output:0*
T0*"
_output_shapes
:?
pembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/BroadcastTo/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
jembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/BroadcastToBroadcastTo?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Sum:output:0yembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/BroadcastTo/shape:output:0*
T0*"
_output_shapes
:?
tembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
bembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/SumSumsembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/BroadcastTo:output:0}embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Sum/reduction_indices:output:0*
T0*
_output_shapes
:?
dembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
bembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/NegNegmembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/zeros:output:0*
T0*
_output_shapes
: ?
dembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Neg_1Negfembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Neg:y:0*
T0*
_output_shapes
: ?
cembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/onesConst*
_output_shapes

:*
dtype0*
valueB*  ???
bembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/mulMullembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/ones:output:0hembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Neg_1:y:0*
T0*
_output_shapes

:?
tembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"?????????
bembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/SumSumfembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/mul:z:0}embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Sum/reduction_indices:output:0*
T0*
_output_shapes
: ?
Aembedding_variational/reshapeSampleIndependentNormal/log_prob/subSubkembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Sum:output:0kembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Sum:output:0*
T0*
_output_shapes
:?
embedding_variational/subSubbembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Sum:output:0Eembedding_variational/reshapeSampleIndependentNormal/log_prob/sub:z:0*
T0*
_output_shapes
:e
embedding_variational/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
embedding_variational/MeanMeanembedding_variational/sub:z:0$embedding_variational/Const:output:0*
T0*
_output_shapes
: `
embedding_variational/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *x?V9?
embedding_variational/mulMul#embedding_variational/Mean:output:0$embedding_variational/mul/y:output:0*
T0*
_output_shapes
: ?
*embedding_variational/mul_1/ReadVariableOpReadVariableOp3embedding_variational_mul_1_readvariableop_resource*
_output_shapes
: *
dtype0?
embedding_variational/mul_1Mulembedding_variational/mul:z:02embedding_variational/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
: J
packed/0Const*
_output_shapes
: *
dtype0*
value	B :`
packedPackpacked/0:output:0packed_values_1*
N*
T0*
_output_shapes
:Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitVJembedding_variational/embedding_normal_diag/Normal/sample/Reshape:output:0packed:output:0split/split_dim:output:0*
T0*

Tlen0*J
_output_shapes8
6:?????????:?????????*
	num_split}
distance_based/minkowski/subSubsplit:output:0split:output:1*
T0*/
_output_shapes
:?????????n
distance_based/minkowski/ShapeShape distance_based/minkowski/sub:z:0*
T0*
_output_shapes
:v
,distance_based/minkowski/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
.distance_based/minkowski/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????x
.distance_based/minkowski/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&distance_based/minkowski/strided_sliceStridedSlice'distance_based/minkowski/Shape:output:05distance_based/minkowski/strided_slice/stack:output:07distance_based/minkowski/strided_slice/stack_1:output:07distance_based/minkowski/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:h
#distance_based/minkowski/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
distance_based/minkowski/onesFill/distance_based/minkowski/strided_slice:output:0,distance_based/minkowski/ones/Const:output:0*
T0*+
_output_shapes
:??????????
'distance_based/minkowski/ReadVariableOpReadVariableOp0distance_based_minkowski_readvariableop_resource*
_output_shapes
: *
dtype0?
distance_based/minkowski/mulMul/distance_based/minkowski/ReadVariableOp:value:0&distance_based/minkowski/ones:output:0*
T0*+
_output_shapes
:??????????
3distance_based/minkowski/BroadcastTo/ReadVariableOpReadVariableOp<distance_based_minkowski_broadcastto_readvariableop_resource*
_output_shapes
:*
dtype0?
$distance_based/minkowski/BroadcastToBroadcastTo;distance_based/minkowski/BroadcastTo/ReadVariableOp:value:0'distance_based/minkowski/Shape:output:0*
T0*/
_output_shapes
:?????????
distance_based/minkowski/AbsAbs distance_based/minkowski/sub:z:0*
T0*/
_output_shapes
:?????????r
'distance_based/minkowski/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
#distance_based/minkowski/ExpandDims
ExpandDims distance_based/minkowski/mul:z:00distance_based/minkowski/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
distance_based/minkowski/PowPow distance_based/minkowski/Abs:y:0,distance_based/minkowski/ExpandDims:output:0*
T0*/
_output_shapes
:??????????
distance_based/minkowski/Mul_1Mul distance_based/minkowski/Pow:z:0-distance_based/minkowski/BroadcastTo:output:0*
T0*/
_output_shapes
:?????????y
.distance_based/minkowski/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
distance_based/minkowski/SumSum"distance_based/minkowski/Mul_1:z:07distance_based/minkowski/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(g
"distance_based/minkowski/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
 distance_based/minkowski/truedivRealDiv+distance_based/minkowski/truediv/x:output:0,distance_based/minkowski/ExpandDims:output:0*
T0*/
_output_shapes
:??????????
distance_based/minkowski/Pow_1Pow%distance_based/minkowski/Sum:output:0$distance_based/minkowski/truediv:z:0*
T0*/
_output_shapes
:??????????
!distance_based/minkowski/IdentityIdentity"distance_based/minkowski/Pow_1:z:0*
T0*/
_output_shapes
:??????????
"distance_based/minkowski/IdentityN	IdentityN"distance_based/minkowski/Pow_1:z:0 distance_based/minkowski/sub:z:0-distance_based/minkowski/BroadcastTo:output:0 distance_based/minkowski/mul:z:0*
T
2*,
_gradient_op_typeCustomGradient-119451*|
_output_shapesj
h:?????????:?????????:?????????:??????????
 distance_based/minkowski/SqueezeSqueeze+distance_based/minkowski/IdentityN:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

??????????
8distance_based/exponential_similarity/Neg/ReadVariableOpReadVariableOpAdistance_based_exponential_similarity_neg_readvariableop_resource*
_output_shapes
: *
dtype0?
)distance_based/exponential_similarity/NegNeg@distance_based/exponential_similarity/Neg/ReadVariableOp:value:0*
T0*
_output_shapes
: ?
8distance_based/exponential_similarity/Pow/ReadVariableOpReadVariableOpAdistance_based_exponential_similarity_pow_readvariableop_resource*
_output_shapes
: *
dtype0?
)distance_based/exponential_similarity/PowPow)distance_based/minkowski/Squeeze:output:0@distance_based/exponential_similarity/Pow/ReadVariableOp:value:0*
T0*+
_output_shapes
:??????????
)distance_based/exponential_similarity/mulMul-distance_based/exponential_similarity/Neg:y:0-distance_based/exponential_similarity/Pow:z:0*
T0*+
_output_shapes
:??????????
)distance_based/exponential_similarity/ExpExp-distance_based/exponential_similarity/mul:z:0*
T0*+
_output_shapes
:??????????
8distance_based/exponential_similarity/add/ReadVariableOpReadVariableOpAdistance_based_exponential_similarity_add_readvariableop_resource*
_output_shapes
: *
dtype0?
)distance_based/exponential_similarity/addAddV2-distance_based/exponential_similarity/Exp:y:0@distance_based/exponential_similarity/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????_
CastCastNotEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:??????????
rank_sim_zero_out_nonpresentMul-distance_based/exponential_similarity/add:z:0Cast:y:0*
T0*+
_output_shapes
:??????????

GatherV2_1GatherV2 rank_sim_zero_out_nonpresent:z:0gatherv2_1_indicesgatherv2_axis*
Taxis0*
Tindices0*
Tparams0*/
_output_shapes
:?????????P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :u

ExpandDims
ExpandDimsCast:y:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????G
ConstConst*
_output_shapes
: *
dtype0*
value	B : ?

GatherV2_2GatherV2ExpandDims:output:0Const:output:0gatherv2_axis*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:?????????M
Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B :?
CumsumCumsumGatherV2_1:output:0Cumsum/axis:output:0*
T0*/
_output_shapes
:?????????*
reverse(N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3u
MaximumMaximumGatherV2_1:output:0Maximum/y:output:0*
T0*/
_output_shapes
:?????????P
Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3r
	Maximum_1MaximumCumsum:out:0Maximum_1/y:output:0*
T0*/
_output_shapes
:?????????Q
LogLogMaximum:z:0*
T0*/
_output_shapes
:?????????U
Log_1LogMaximum_1:z:0*
T0*/
_output_shapes
:?????????X
subSubLog:y:0	Log_1:y:0*
T0*/
_output_shapes
:?????????T
mulMulmul_xsub:z:0*
T0*/
_output_shapes
:?????????W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :i
SumSummul:z:0Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????N
ExpExpSum:output:0*
T0*+
_output_shapes
:?????????`
mul_1MulGatherV2_2:output:0Exp:y:0*
T0*+
_output_shapes
:?????????Y
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
Sum_1Sum	mul_1:z:0 Sum_1/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(L
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *    f
EqualEqualSum_1:output:0Equal/y:output:0*
T0*+
_output_shapes
:?????????^
Cast_1Cast	Equal:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????_
truedivRealDiv
Cast_1:y:0	truediv_y*
T0*+
_output_shapes
:?????????Z
addAddV2	mul_1:z:0truediv:z:0*
T0*+
_output_shapes
:?????????`
add_1AddV2Sum_1:output:0
Cast_1:y:0*
T0*+
_output_shapes
:?????????^
	truediv_1RealDivadd:z:0	add_1:z:0*
T0*+
_output_shapes
:?????????`
IdentityIdentitytruediv_1:z:0^NoOp*
T0*+
_output_shapes
:?????????_

Identity_1Identityembedding_variational/mul_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp9^distance_based/exponential_similarity/Neg/ReadVariableOp9^distance_based/exponential_similarity/Pow/ReadVariableOp9^distance_based/exponential_similarity/add/ReadVariableOp4^distance_based/minkowski/BroadcastTo/ReadVariableOp(^distance_based/minkowski/ReadVariableOpu^embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOp~^embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOpq^embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOpz^embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOp=^embedding_variational/embedding_normal_diag/embedding_lookupW^embedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOpP^embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookupj^embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOp+^embedding_variational/mul_1/ReadVariableOp?^embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp?^embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:?????????:?????????:: : : : : : : : : : : : ::: 2t
8distance_based/exponential_similarity/Neg/ReadVariableOp8distance_based/exponential_similarity/Neg/ReadVariableOp2t
8distance_based/exponential_similarity/Pow/ReadVariableOp8distance_based/exponential_similarity/Pow/ReadVariableOp2t
8distance_based/exponential_similarity/add/ReadVariableOp8distance_based/exponential_similarity/add/ReadVariableOp2j
3distance_based/minkowski/BroadcastTo/ReadVariableOp3distance_based/minkowski/BroadcastTo/ReadVariableOp2R
'distance_based/minkowski/ReadVariableOp'distance_based/minkowski/ReadVariableOp2?
tembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOptembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOp2?
}embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOp}embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOp2?
pembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOppembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOp2?
yembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOpyembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOp2|
<embedding_variational/embedding_normal_diag/embedding_lookup<embedding_variational/embedding_normal_diag/embedding_lookup2?
Vembedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOpVembedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOp2?
Oembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookupOembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup2?
iembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOpiembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOp2X
*embedding_variational/mul_1/ReadVariableOp*embedding_variational/mul_1/ReadVariableOp2?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp2?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp:g c
+
_output_shapes
:?????????
4
_user_specified_nameinputs/2rank1/stimulus_set:\X
+
_output_shapes
:?????????
)
_user_specified_nameinputs/agent_id: 

_output_shapes
::

_output_shapes
: :	

_output_shapes
: :$ 

_output_shapes

::,(
&
_output_shapes
::

_output_shapes
: 
??
?
K__inference_rank_similarity_layer_call_and_return_conditional_losses_119808
inputs_2rank1_stimulus_set
inputs_agent_id
gatherv2_indices
gatherv2_axisU
Cembedding_variational_embedding_normal_diag_embedding_lookup_119523:q
_embedding_variational_embedding_normal_diag_embedding_lookup_1_softplus_readvariableop_resource:h
Vembedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_119572:?
rembedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_1_softplus_readvariableop_resource:=
3embedding_variational_mul_1_readvariableop_resource: 
packed_values_1:
0distance_based_minkowski_readvariableop_resource: J
<distance_based_minkowski_broadcastto_readvariableop_resource:K
Adistance_based_exponential_similarity_neg_readvariableop_resource: K
Adistance_based_exponential_similarity_pow_readvariableop_resource: K
Adistance_based_exponential_similarity_add_readvariableop_resource: 
gatherv2_1_indices	
mul_x
	truediv_y
identity

identity_1??8distance_based/exponential_similarity/Neg/ReadVariableOp?8distance_based/exponential_similarity/Pow/ReadVariableOp?8distance_based/exponential_similarity/add/ReadVariableOp?3distance_based/minkowski/BroadcastTo/ReadVariableOp?'distance_based/minkowski/ReadVariableOp?tembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOp?}embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOp?pembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOp?yembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOp?<embedding_variational/embedding_normal_diag/embedding_lookup?Vembedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOp?Oembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup?iembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOp?*embedding_variational/mul_1/ReadVariableOp??embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp??embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp?
GatherV2GatherV2inputs_2rank1_stimulus_setgatherv2_indicesgatherv2_axis*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:?????????L

NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : r
NotEqualNotEqualGatherV2:output:0NotEqual/y:output:0*
T0*+
_output_shapes
:??????????
<embedding_variational/embedding_normal_diag/embedding_lookupResourceGatherCembedding_variational_embedding_normal_diag_embedding_lookup_119523inputs_2rank1_stimulus_set*
Tindices0*V
_classL
JHloc:@embedding_variational/embedding_normal_diag/embedding_lookup/119523*/
_output_shapes
:?????????*
dtype0?
Eembedding_variational/embedding_normal_diag/embedding_lookup/IdentityIdentityEembedding_variational/embedding_normal_diag/embedding_lookup:output:0*
T0*V
_classL
JHloc:@embedding_variational/embedding_normal_diag/embedding_lookup/119523*/
_output_shapes
:??????????
Gembedding_variational/embedding_normal_diag/embedding_lookup/Identity_1IdentityNembedding_variational/embedding_normal_diag/embedding_lookup/Identity:output:0*
T0*/
_output_shapes
:??????????
Vembedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOpReadVariableOp_embedding_variational_embedding_normal_diag_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
Gembedding_variational/embedding_normal_diag/embedding_lookup_1/SoftplusSoftplus^embedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
Dembedding_variational/embedding_normal_diag/embedding_lookup_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
Bembedding_variational/embedding_normal_diag/embedding_lookup_1/addAddV2Membedding_variational/embedding_normal_diag/embedding_lookup_1/add/x:output:0Uembedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus:activations:0*
T0*
_output_shapes

:?
Cembedding_variational/embedding_normal_diag/embedding_lookup_1/axisConst*U
_classK
IGloc:@embedding_variational/embedding_normal_diag/embedding_lookup_1/add*
_output_shapes
: *
dtype0*
value	B : ?
>embedding_variational/embedding_normal_diag/embedding_lookup_1GatherV2Fembedding_variational/embedding_normal_diag/embedding_lookup_1/add:z:0inputs_2rank1_stimulus_setLembedding_variational/embedding_normal_diag/embedding_lookup_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*U
_classK
IGloc:@embedding_variational/embedding_normal_diag/embedding_lookup_1/add*/
_output_shapes
:??????????
Gembedding_variational/embedding_normal_diag/embedding_lookup_1/IdentityIdentityGembedding_variational/embedding_normal_diag/embedding_lookup_1:output:0*
T0*/
_output_shapes
:??????????
Fembedding_variational/embedding_normal_diag/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
?embedding_variational/embedding_normal_diag/Normal/sample/ShapeShapePembedding_variational/embedding_normal_diag/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
?embedding_variational/embedding_normal_diag/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
Membedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Oembedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Oembedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Gembedding_variational/embedding_normal_diag/Normal/sample/strided_sliceStridedSliceHembedding_variational/embedding_normal_diag/Normal/sample/Shape:output:0Vembedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack:output:0Xembedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack_1:output:0Xembedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Aembedding_variational/embedding_normal_diag/Normal/sample/Shape_1ShapePembedding_variational/embedding_normal_diag/embedding_lookup_1/Identity:output:0*
T0*
_output_shapes
:?
Aembedding_variational/embedding_normal_diag/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : ?
Oembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Qembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Qembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Iembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1StridedSliceJembedding_variational/embedding_normal_diag/Normal/sample/Shape_1:output:0Xembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack:output:0Zembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack_1:output:0Zembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Jembedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
Lembedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
Gembedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgsBroadcastArgsUembedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs/s0_1:output:0Pembedding_variational/embedding_normal_diag/Normal/sample/strided_slice:output:0*
_output_shapes
:?
Iembedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs_1BroadcastArgsLembedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs:r0:0Rembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1:output:0*
_output_shapes
:?
Iembedding_variational/embedding_normal_diag/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
Eembedding_variational/embedding_normal_diag/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
@embedding_variational/embedding_normal_diag/Normal/sample/concatConcatV2Rembedding_variational/embedding_normal_diag/Normal/sample/concat/values_0:output:0Nembedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs_1:r0:0Nembedding_variational/embedding_normal_diag/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Sembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
Uembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
cembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalIembedding_variational/embedding_normal_diag/Normal/sample/concat:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????*
dtype0?
Rembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/mulMullembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/RandomStandardNormal:output:0^embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/stddev:output:0*
T0*N
_output_shapes<
::8?????????????????????????????????????
Nembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normalAddV2Vembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/mul:z:0\embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/mean:output:0*
T0*N
_output_shapes<
::8?????????????????????????????????????
=embedding_variational/embedding_normal_diag/Normal/sample/mulMulRembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal:z:0Pembedding_variational/embedding_normal_diag/embedding_lookup_1/Identity:output:0*
T0*E
_output_shapes3
1:/????????????????????????????
=embedding_variational/embedding_normal_diag/Normal/sample/addAddV2Aembedding_variational/embedding_normal_diag/Normal/sample/mul:z:0Pembedding_variational/embedding_normal_diag/embedding_lookup/Identity_1:output:0*
T0*E
_output_shapes3
1:/????????????????????????????
Aembedding_variational/embedding_normal_diag/Normal/sample/Shape_2ShapeAembedding_variational/embedding_normal_diag/Normal/sample/add:z:0*
T0*
_output_shapes
:?
Oembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Qembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Qembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Iembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2StridedSliceJembedding_variational/embedding_normal_diag/Normal/sample/Shape_2:output:0Xembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack:output:0Zembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack_1:output:0Zembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Gembedding_variational/embedding_normal_diag/Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Bembedding_variational/embedding_normal_diag/Normal/sample/concat_1ConcatV2Oembedding_variational/embedding_normal_diag/Normal/sample/sample_shape:output:0Rembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2:output:0Pembedding_variational/embedding_normal_diag/Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Aembedding_variational/embedding_normal_diag/Normal/sample/ReshapeReshapeAembedding_variational/embedding_normal_diag/Normal/sample/add:z:0Kembedding_variational/embedding_normal_diag/Normal/sample/concat_1:output:0*
T0*/
_output_shapes
:??????????
1embedding_variational/embedding_shared/zeros_like	ZerosLikeinputs_2rank1_stimulus_set*
T0*+
_output_shapes
:??????????
Oembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookupResourceGatherVembedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_1195725embedding_variational/embedding_shared/zeros_like:y:0*
Tindices0*i
_class_
][loc:@embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/119572*/
_output_shapes
:?????????*
dtype0?
Xembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/IdentityIdentityXembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup:output:0*
T0*i
_class_
][loc:@embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/119572*/
_output_shapes
:??????????
Zembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/Identity_1Identityaembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/Identity:output:0*
T0*/
_output_shapes
:??????????
iembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOpReadVariableOprembedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
Zembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/SoftplusSoftplusqembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
Wembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
Uembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/addAddV2`embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add/x:output:0hembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus:activations:0*
T0*
_output_shapes

:?
Vembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/axisConst*h
_class^
\Zloc:@embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add*
_output_shapes
: *
dtype0*
value	B : ?
Qembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1GatherV2Yembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add:z:05embedding_variational/embedding_shared/zeros_like:y:0_embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*h
_class^
\Zloc:@embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add*/
_output_shapes
:??????????
Zembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/IdentityIdentityZembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1:output:0*
T0*/
_output_shapes
:??????????
Yembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Rembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/ShapeShapecembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
Rembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
`embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
bembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
bembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Zembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_sliceStridedSlice[embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape:output:0iembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack:output:0kembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack_1:output:0kembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Tembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape_1Shapecembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Identity:output:0*
T0*
_output_shapes
:?
Tembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : ?
bembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
dembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
dembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1StridedSlice]embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape_1:output:0kembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack:output:0membedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack_1:output:0membedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
]embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
_embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
Zembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgsBroadcastArgshembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs/s0_1:output:0cembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice:output:0*
_output_shapes
:?
\embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs_1BroadcastArgs_embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs:r0:0eembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1:output:0*
_output_shapes
:?
\embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
Xembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Sembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concatConcatV2eembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat/values_0:output:0aembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs_1:r0:0aembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
fembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
hembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
vembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal\embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????*
dtype0?
eembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/mulMulembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/RandomStandardNormal:output:0qembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/stddev:output:0*
T0*N
_output_shapes<
::8?????????????????????????????????????
aembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normalAddV2iembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/mul:z:0oembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/mean:output:0*
T0*N
_output_shapes<
::8?????????????????????????????????????
Pembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/mulMuleembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal:z:0cembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Identity:output:0*
T0*E
_output_shapes3
1:/????????????????????????????
Pembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/addAddV2Tembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/mul:z:0cembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/Identity_1:output:0*
T0*E
_output_shapes3
1:/????????????????????????????
Tembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape_2ShapeTembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/add:z:0*
T0*
_output_shapes
:?
bembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
dembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2StridedSlice]embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape_2:output:0kembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack:output:0membedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack_1:output:0membedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Zembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Uembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat_1ConcatV2bembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/sample_shape:output:0eembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2:output:0cembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Tembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/ReshapeReshapeTembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/add:z:0^embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat_1:output:0*
T0*/
_output_shapes
:?????????p
-embedding_variational/reshape/event_shape_outConst*
_output_shapes
: *
dtype0*
valueB }
,embedding_variational/reshape/event_shape_inConst*
_output_shapes
:*
dtype0*
valueB"      ?
:embedding_variational/SampleIndependentNormal/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB"      {
9embedding_variational/reshapeSampleIndependentNormal/zeroConst*
_output_shapes
: *
dtype0*
value	B : ?
`embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
qembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
nembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
pembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOpReadVariableOpCembedding_variational_embedding_normal_diag_embedding_lookup_119523*
_output_shapes

:*
dtype0?
yembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOpReadVariableOp_embedding_variational_embedding_normal_diag_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
jembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/SoftplusSoftplus?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
gembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
eembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/addAddV2pembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add/x:output:0xembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus:activations:0*
T0*
_output_shapes

:?
qembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"      ?
gembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
uembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
wembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
wembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
oembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_sliceStridedSlicezembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/shape_as_tensor:output:0~embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack:output:0?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack_1:output:0?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
sembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"      ?
iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : ?
wembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
yembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
yembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
qembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1StridedSlice|embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/shape_as_tensor_1:output:0?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack:output:0?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack_1:output:0?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
rembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
tembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
oembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgsBroadcastArgs}embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs/s0_1:output:0xembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice:output:0*
_output_shapes
:?
qembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs_1BroadcastArgstembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs:r0:0zembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1:output:0*
_output_shapes
:?
qembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
membedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
hembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concatConcatV2zembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat/values_0:output:0vembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs_1:r0:0vembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
{embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
}embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalqembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat:output:0*
T0*"
_output_shapes
:*
dtype0?
zembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/mulMul?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/RandomStandardNormal:output:0?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/stddev:output:0*
T0*"
_output_shapes
:?
vembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normalAddV2~embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/mul:z:0?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/mean:output:0*
T0*"
_output_shapes
:?
eembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/mulMulzembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal:z:0iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add:z:0*
T0*"
_output_shapes
:?
gembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add_1AddV2iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/mul:z:0xembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOp:value:0*
T0*"
_output_shapes
:?
oembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReshapeReshapekembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add_1:z:0xembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Reshape/shape:output:0*
T0*"
_output_shapes
:?
aembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
[embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/ReshapeReshaperembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Reshape:output:0jembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Reshape/shape:output:0*
T0*"
_output_shapes
:?
}embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOpReadVariableOp_embedding_variational_embedding_normal_diag_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
nembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/SoftplusSoftplus?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
kembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/addAddV2tembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add/x:output:0|embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus:activations:0*
T0*
_output_shapes

:?
membedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/truedivRealDivdembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Reshape:output:0membedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add:z:0*
T0*"
_output_shapes
:?
tembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOpReadVariableOpCembedding_variational_embedding_normal_diag_embedding_lookup_119523*
_output_shapes

:*
dtype0?
oembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/truediv_1RealDiv|embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOp:value:0membedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add:z:0*
T0*
_output_shapes

:?
wembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/SquaredDifferenceSquaredDifferenceqembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/truediv:z:0sembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/truediv_1:z:0*
T0*"
_output_shapes
:?
kembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/mulMultembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/mul/x:output:0{embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/SquaredDifference:z:0*
T0*"
_output_shapes
:?
kembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??k??
iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/LogLogmembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add:z:0*
T0*
_output_shapes

:?
kembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add_1AddV2tembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Const:output:0membedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Log:y:0*
T0*
_output_shapes

:?
iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/subSubmembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/mul:z:0oembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add_1:z:0*
T0*"
_output_shapes
:?
kembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"?????????
Yembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/SumSummembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/sub:z:0tembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Sum/reduction_indices:output:0*
T0*
_output_shapes
:?
Zembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/inverse/output_shapeConst*
_output_shapes
:*
dtype0*)
value B"               ?
Uembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/inverse/ReshapeReshapedembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Reshape:output:0cembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/inverse/output_shape:output:0*
T0**
_output_shapes
:?
nembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/event_shape_tensor/ConstConst*
_output_shapes
:*
dtype0*%
valueB"            ?
tembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/event_shape_tensor/event_shapeIdentitywembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/event_shape_tensor/Const:output:0*
T0*
_output_shapes
:?
iembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_event_shape_tensor/IdentityIdentity}embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/event_shape_tensor/event_shape:output:0*
T0*
_output_shapes
:?
membedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_event_shape_tensor/output_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
lembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"               ?
fembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/ReshapeReshape^embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/inverse/Reshape:output:0uembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Reshape/shape:output:0*
T0**
_output_shapes
:?
membedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                ?
hembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/transpose	Transposeoembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Reshape:output:0vembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/transpose/perm:output:0*
T0**
_output_shapes
:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOpReadVariableOprembedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/SoftplusSoftplus?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/addAddV2?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add/x:output:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus:activations:0*
T0*
_output_shapes

:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/truedivRealDivlembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/transpose:y:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add:z:0*
T0**
_output_shapes
:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOpReadVariableOpVembedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_119572*
_output_shapes

:*
dtype0?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/truediv_1RealDiv?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp:value:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add:z:0*
T0*
_output_shapes

:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/SquaredDifferenceSquaredDifference?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/truediv:z:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/truediv_1:z:0*
T0**
_output_shapes
:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/mulMul?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/mul/x:output:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/SquaredDifference:z:0*
T0**
_output_shapes
:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??k??
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/LogLog?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add:z:0*
T0*
_output_shapes

:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add_1AddV2?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Const:output:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Log:y:0*
T0*
_output_shapes

:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/subSub?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/mul:z:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add_1:z:0*
T0**
_output_shapes
:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"?????????
}embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/SumSum?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/sub:z:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Sum/reduction_indices:output:0*
T0*"
_output_shapes
:?
pembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/BroadcastTo/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
jembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/BroadcastToBroadcastTo?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Sum:output:0yembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/BroadcastTo/shape:output:0*
T0*"
_output_shapes
:?
tembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
bembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/SumSumsembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/BroadcastTo:output:0}embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Sum/reduction_indices:output:0*
T0*
_output_shapes
:?
dembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
bembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/NegNegmembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/zeros:output:0*
T0*
_output_shapes
: ?
dembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Neg_1Negfembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Neg:y:0*
T0*
_output_shapes
: ?
cembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/onesConst*
_output_shapes

:*
dtype0*
valueB*  ???
bembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/mulMullembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/ones:output:0hembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Neg_1:y:0*
T0*
_output_shapes

:?
tembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"?????????
bembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/SumSumfembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/mul:z:0}embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Sum/reduction_indices:output:0*
T0*
_output_shapes
: ?
Aembedding_variational/reshapeSampleIndependentNormal/log_prob/subSubkembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Sum:output:0kembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Sum:output:0*
T0*
_output_shapes
:?
embedding_variational/subSubbembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Sum:output:0Eembedding_variational/reshapeSampleIndependentNormal/log_prob/sub:z:0*
T0*
_output_shapes
:e
embedding_variational/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
embedding_variational/MeanMeanembedding_variational/sub:z:0$embedding_variational/Const:output:0*
T0*
_output_shapes
: `
embedding_variational/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *x?V9?
embedding_variational/mulMul#embedding_variational/Mean:output:0$embedding_variational/mul/y:output:0*
T0*
_output_shapes
: ?
*embedding_variational/mul_1/ReadVariableOpReadVariableOp3embedding_variational_mul_1_readvariableop_resource*
_output_shapes
: *
dtype0?
embedding_variational/mul_1Mulembedding_variational/mul:z:02embedding_variational/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
: J
packed/0Const*
_output_shapes
: *
dtype0*
value	B :`
packedPackpacked/0:output:0packed_values_1*
N*
T0*
_output_shapes
:Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitVJembedding_variational/embedding_normal_diag/Normal/sample/Reshape:output:0packed:output:0split/split_dim:output:0*
T0*

Tlen0*J
_output_shapes8
6:?????????:?????????*
	num_split}
distance_based/minkowski/subSubsplit:output:0split:output:1*
T0*/
_output_shapes
:?????????n
distance_based/minkowski/ShapeShape distance_based/minkowski/sub:z:0*
T0*
_output_shapes
:v
,distance_based/minkowski/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
.distance_based/minkowski/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????x
.distance_based/minkowski/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&distance_based/minkowski/strided_sliceStridedSlice'distance_based/minkowski/Shape:output:05distance_based/minkowski/strided_slice/stack:output:07distance_based/minkowski/strided_slice/stack_1:output:07distance_based/minkowski/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:h
#distance_based/minkowski/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
distance_based/minkowski/onesFill/distance_based/minkowski/strided_slice:output:0,distance_based/minkowski/ones/Const:output:0*
T0*+
_output_shapes
:??????????
'distance_based/minkowski/ReadVariableOpReadVariableOp0distance_based_minkowski_readvariableop_resource*
_output_shapes
: *
dtype0?
distance_based/minkowski/mulMul/distance_based/minkowski/ReadVariableOp:value:0&distance_based/minkowski/ones:output:0*
T0*+
_output_shapes
:??????????
3distance_based/minkowski/BroadcastTo/ReadVariableOpReadVariableOp<distance_based_minkowski_broadcastto_readvariableop_resource*
_output_shapes
:*
dtype0?
$distance_based/minkowski/BroadcastToBroadcastTo;distance_based/minkowski/BroadcastTo/ReadVariableOp:value:0'distance_based/minkowski/Shape:output:0*
T0*/
_output_shapes
:?????????
distance_based/minkowski/AbsAbs distance_based/minkowski/sub:z:0*
T0*/
_output_shapes
:?????????r
'distance_based/minkowski/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
#distance_based/minkowski/ExpandDims
ExpandDims distance_based/minkowski/mul:z:00distance_based/minkowski/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
distance_based/minkowski/PowPow distance_based/minkowski/Abs:y:0,distance_based/minkowski/ExpandDims:output:0*
T0*/
_output_shapes
:??????????
distance_based/minkowski/Mul_1Mul distance_based/minkowski/Pow:z:0-distance_based/minkowski/BroadcastTo:output:0*
T0*/
_output_shapes
:?????????y
.distance_based/minkowski/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
distance_based/minkowski/SumSum"distance_based/minkowski/Mul_1:z:07distance_based/minkowski/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(g
"distance_based/minkowski/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
 distance_based/minkowski/truedivRealDiv+distance_based/minkowski/truediv/x:output:0,distance_based/minkowski/ExpandDims:output:0*
T0*/
_output_shapes
:??????????
distance_based/minkowski/Pow_1Pow%distance_based/minkowski/Sum:output:0$distance_based/minkowski/truediv:z:0*
T0*/
_output_shapes
:??????????
!distance_based/minkowski/IdentityIdentity"distance_based/minkowski/Pow_1:z:0*
T0*/
_output_shapes
:??????????
"distance_based/minkowski/IdentityN	IdentityN"distance_based/minkowski/Pow_1:z:0 distance_based/minkowski/sub:z:0-distance_based/minkowski/BroadcastTo:output:0 distance_based/minkowski/mul:z:0*
T
2*,
_gradient_op_typeCustomGradient-119745*|
_output_shapesj
h:?????????:?????????:?????????:??????????
 distance_based/minkowski/SqueezeSqueeze+distance_based/minkowski/IdentityN:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

??????????
8distance_based/exponential_similarity/Neg/ReadVariableOpReadVariableOpAdistance_based_exponential_similarity_neg_readvariableop_resource*
_output_shapes
: *
dtype0?
)distance_based/exponential_similarity/NegNeg@distance_based/exponential_similarity/Neg/ReadVariableOp:value:0*
T0*
_output_shapes
: ?
8distance_based/exponential_similarity/Pow/ReadVariableOpReadVariableOpAdistance_based_exponential_similarity_pow_readvariableop_resource*
_output_shapes
: *
dtype0?
)distance_based/exponential_similarity/PowPow)distance_based/minkowski/Squeeze:output:0@distance_based/exponential_similarity/Pow/ReadVariableOp:value:0*
T0*+
_output_shapes
:??????????
)distance_based/exponential_similarity/mulMul-distance_based/exponential_similarity/Neg:y:0-distance_based/exponential_similarity/Pow:z:0*
T0*+
_output_shapes
:??????????
)distance_based/exponential_similarity/ExpExp-distance_based/exponential_similarity/mul:z:0*
T0*+
_output_shapes
:??????????
8distance_based/exponential_similarity/add/ReadVariableOpReadVariableOpAdistance_based_exponential_similarity_add_readvariableop_resource*
_output_shapes
: *
dtype0?
)distance_based/exponential_similarity/addAddV2-distance_based/exponential_similarity/Exp:y:0@distance_based/exponential_similarity/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????_
CastCastNotEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:??????????
rank_sim_zero_out_nonpresentMul-distance_based/exponential_similarity/add:z:0Cast:y:0*
T0*+
_output_shapes
:??????????

GatherV2_1GatherV2 rank_sim_zero_out_nonpresent:z:0gatherv2_1_indicesgatherv2_axis*
Taxis0*
Tindices0*
Tparams0*/
_output_shapes
:?????????P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :u

ExpandDims
ExpandDimsCast:y:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????G
ConstConst*
_output_shapes
: *
dtype0*
value	B : ?

GatherV2_2GatherV2ExpandDims:output:0Const:output:0gatherv2_axis*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:?????????M
Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B :?
CumsumCumsumGatherV2_1:output:0Cumsum/axis:output:0*
T0*/
_output_shapes
:?????????*
reverse(N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3u
MaximumMaximumGatherV2_1:output:0Maximum/y:output:0*
T0*/
_output_shapes
:?????????P
Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3r
	Maximum_1MaximumCumsum:out:0Maximum_1/y:output:0*
T0*/
_output_shapes
:?????????Q
LogLogMaximum:z:0*
T0*/
_output_shapes
:?????????U
Log_1LogMaximum_1:z:0*
T0*/
_output_shapes
:?????????X
subSubLog:y:0	Log_1:y:0*
T0*/
_output_shapes
:?????????T
mulMulmul_xsub:z:0*
T0*/
_output_shapes
:?????????W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :i
SumSummul:z:0Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????N
ExpExpSum:output:0*
T0*+
_output_shapes
:?????????`
mul_1MulGatherV2_2:output:0Exp:y:0*
T0*+
_output_shapes
:?????????Y
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
Sum_1Sum	mul_1:z:0 Sum_1/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(L
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *    f
EqualEqualSum_1:output:0Equal/y:output:0*
T0*+
_output_shapes
:?????????^
Cast_1Cast	Equal:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????_
truedivRealDiv
Cast_1:y:0	truediv_y*
T0*+
_output_shapes
:?????????Z
addAddV2	mul_1:z:0truediv:z:0*
T0*+
_output_shapes
:?????????`
add_1AddV2Sum_1:output:0
Cast_1:y:0*
T0*+
_output_shapes
:?????????^
	truediv_1RealDivadd:z:0	add_1:z:0*
T0*+
_output_shapes
:?????????`
IdentityIdentitytruediv_1:z:0^NoOp*
T0*+
_output_shapes
:?????????_

Identity_1Identityembedding_variational/mul_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp9^distance_based/exponential_similarity/Neg/ReadVariableOp9^distance_based/exponential_similarity/Pow/ReadVariableOp9^distance_based/exponential_similarity/add/ReadVariableOp4^distance_based/minkowski/BroadcastTo/ReadVariableOp(^distance_based/minkowski/ReadVariableOpu^embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOp~^embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOpq^embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOpz^embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOp=^embedding_variational/embedding_normal_diag/embedding_lookupW^embedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOpP^embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookupj^embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOp+^embedding_variational/mul_1/ReadVariableOp?^embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp?^embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:?????????:?????????:: : : : : : : : : : : : ::: 2t
8distance_based/exponential_similarity/Neg/ReadVariableOp8distance_based/exponential_similarity/Neg/ReadVariableOp2t
8distance_based/exponential_similarity/Pow/ReadVariableOp8distance_based/exponential_similarity/Pow/ReadVariableOp2t
8distance_based/exponential_similarity/add/ReadVariableOp8distance_based/exponential_similarity/add/ReadVariableOp2j
3distance_based/minkowski/BroadcastTo/ReadVariableOp3distance_based/minkowski/BroadcastTo/ReadVariableOp2R
'distance_based/minkowski/ReadVariableOp'distance_based/minkowski/ReadVariableOp2?
tembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOptembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOp2?
}embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOp}embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOp2?
pembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOppembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOp2?
yembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOpyembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOp2|
<embedding_variational/embedding_normal_diag/embedding_lookup<embedding_variational/embedding_normal_diag/embedding_lookup2?
Vembedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOpVembedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOp2?
Oembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookupOembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup2?
iembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOpiembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOp2X
*embedding_variational/mul_1/ReadVariableOp*embedding_variational/mul_1/ReadVariableOp2?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp2?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp:g c
+
_output_shapes
:?????????
4
_user_specified_nameinputs/2rank1/stimulus_set:\X
+
_output_shapes
:?????????
)
_user_specified_nameinputs/agent_id: 

_output_shapes
::

_output_shapes
: :	

_output_shapes
: :$ 

_output_shapes

::,(
&
_output_shapes
::

_output_shapes
: 
?
?
U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_118390
rank1_stimulus_set
agent_id
rank_similarity_118354
rank_similarity_118356(
rank_similarity_118358:(
rank_similarity_118360:(
rank_similarity_118362:(
rank_similarity_118364: 
rank_similarity_118366: 
rank_similarity_118368 
rank_similarity_118370: $
rank_similarity_118372: 
rank_similarity_118374:  
rank_similarity_118376:  
rank_similarity_118378: 
rank_similarity_118380
rank_similarity_118382
rank_similarity_118384
identity

identity_1??'rank_similarity/StatefulPartitionedCall?
'rank_similarity/StatefulPartitionedCallStatefulPartitionedCallrank1_stimulus_setagent_idrank_similarity_118354rank_similarity_118356rank_similarity_118358rank_similarity_118360rank_similarity_118362rank_similarity_118364rank_similarity_118366rank_similarity_118368rank_similarity_118370rank_similarity_118372rank_similarity_118374rank_similarity_118376rank_similarity_118378rank_similarity_118380rank_similarity_118382rank_similarity_118384*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:?????????: *,
_read_only_resource_inputs


*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_rank_similarity_layer_call_and_return_conditional_losses_117747?
IdentityIdentity0rank_similarity/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????p

Identity_1Identity0rank_similarity/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: p
NoOpNoOp(^rank_similarity/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:?????????:?????????:: : : : : : : : : : : : ::: 2R
'rank_similarity/StatefulPartitionedCall'rank_similarity/StatefulPartitionedCall:` \
+
_output_shapes
:?????????
-
_user_specified_name2rank1/stimulus_set:UQ
+
_output_shapes
:?????????
"
_user_specified_name
agent_id: 

_output_shapes
::

_output_shapes
: :	

_output_shapes
: :$ 

_output_shapes

::,(
&
_output_shapes
::

_output_shapes
: 
?e
?
"__inference__traced_restore_120429
file_prefix<
*assignvariableop_embedding_normal_diag_loc:N
<assignvariableop_1_embedding_normal_diag_untransformed_scale:P
>assignvariableop_2_embedding_normal_diag_1_untransformed_scale:&
assignvariableop_3_kl_anneal: @
.assignvariableop_4_embedding_normal_diag_1_loc:*
 assignvariableop_5_minkowski_rho: e
Wassignvariableop_6_stochastic_behavior_model_rank_similarity_distance_based_minkowski_w:7
-assignvariableop_7_exponential_similarity_tau: 9
/assignvariableop_8_exponential_similarity_gamma: 8
.assignvariableop_9_exponential_similarity_beta: '
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: #
assignvariableop_17_total: #
assignvariableop_18_count: F
4assignvariableop_19_adam_embedding_normal_diag_loc_m:V
Dassignvariableop_20_adam_embedding_normal_diag_untransformed_scale_m:X
Fassignvariableop_21_adam_embedding_normal_diag_1_untransformed_scale_m:F
4assignvariableop_22_adam_embedding_normal_diag_loc_v:V
Dassignvariableop_23_adam_embedding_normal_diag_untransformed_scale_v:X
Fassignvariableop_24_adam_embedding_normal_diag_1_untransformed_scale_v:
identity_26??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?

value?
B?
B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp*assignvariableop_embedding_normal_diag_locIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp<assignvariableop_1_embedding_normal_diag_untransformed_scaleIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp>assignvariableop_2_embedding_normal_diag_1_untransformed_scaleIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_kl_annealIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp.assignvariableop_4_embedding_normal_diag_1_locIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp assignvariableop_5_minkowski_rhoIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpWassignvariableop_6_stochastic_behavior_model_rank_similarity_distance_based_minkowski_wIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp-assignvariableop_7_exponential_similarity_tauIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp/assignvariableop_8_exponential_similarity_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_exponential_similarity_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp4assignvariableop_19_adam_embedding_normal_diag_loc_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpDassignvariableop_20_adam_embedding_normal_diag_untransformed_scale_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpFassignvariableop_21_adam_embedding_normal_diag_1_untransformed_scale_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp4assignvariableop_22_adam_embedding_normal_diag_loc_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpDassignvariableop_23_adam_embedding_normal_diag_untransformed_scale_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpFassignvariableop_24_adam_embedding_normal_diag_1_untransformed_scale_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_26IdentityIdentity_25:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_26Identity_26:output:0*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?<
?
#__inference_internal_grad_fn_120299
result_grads_0
result_grads_1
result_grads_2
result_grads_3V
Rmul_stochastic_behavior_model_rank_similarity_distance_based_minkowski_broadcasttoN
Jmul_stochastic_behavior_model_rank_similarity_distance_based_minkowski_subN
Jpow_stochastic_behavior_model_rank_similarity_distance_based_minkowski_absU
Qdiv_no_nan_stochastic_behavior_model_rank_similarity_distance_based_minkowski_powU
Qsub_stochastic_behavior_model_rank_similarity_distance_based_minkowski_expanddimsR
Npow_1_stochastic_behavior_model_rank_similarity_distance_based_minkowski_pow_1W
Sdiv_no_nan_1_stochastic_behavior_model_rank_similarity_distance_based_minkowski_sumR
Nmul_6_stochastic_behavior_model_rank_similarity_distance_based_minkowski_mul_1
identity

identity_1

identity_2?
mulMulRmul_stochastic_behavior_model_rank_similarity_distance_based_minkowski_broadcasttoJmul_stochastic_behavior_model_rank_similarity_distance_based_minkowski_sub*
T0*/
_output_shapes
:?????????J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
powPowJpow_stochastic_behavior_model_rank_similarity_distance_based_minkowski_abspow/y:output:0*
T0*/
_output_shapes
:??????????

div_no_nanDivNoNanQdiv_no_nan_stochastic_behavior_model_rank_similarity_distance_based_minkowski_powpow:z:0*
T0*/
_output_shapes
:?????????_
mul_1Mulmul:z:0div_no_nan:z:0*
T0*/
_output_shapes
:?????????J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
subSubQsub_stochastic_behavior_model_rank_similarity_distance_based_minkowski_expanddimssub/y:output:0*
T0*/
_output_shapes
:??????????
pow_1PowNpow_1_stochastic_behavior_model_rank_similarity_distance_based_minkowski_pow_1sub:z:0*
T0*/
_output_shapes
:?????????J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3a
addAddV2	pow_1:z:0add/y:output:0*
T0*/
_output_shapes
:?????????`
truedivRealDiv	mul_1:z:0add:z:0*
T0*/
_output_shapes
:?????????c
mul_2Mulresult_grads_0truediv:z:0*
T0*/
_output_shapes
:?????????L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
sub_1SubQsub_stochastic_behavior_model_rank_similarity_distance_based_minkowski_expanddimssub_1/y:output:0*
T0*/
_output_shapes
:??????????
pow_2PowNpow_1_stochastic_behavior_model_rank_similarity_distance_based_minkowski_pow_1	sub_1:z:0*
T0*/
_output_shapes
:??????????
mul_3MulQsub_stochastic_behavior_model_rank_similarity_distance_based_minkowski_expanddims	pow_2:z:0*
T0*/
_output_shapes
:?????????L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3e
add_1AddV2	mul_3:z:0add_1/y:output:0*
T0*/
_output_shapes
:??????????
	truediv_1RealDivQdiv_no_nan_stochastic_behavior_model_rank_similarity_distance_based_minkowski_pow	add_1:z:0*
T0*/
_output_shapes
:?????????e
mul_4Mulresult_grads_0truediv_1:z:0*
T0*/
_output_shapes
:?????????P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
	truediv_2RealDivtruediv_2/x:output:0Qsub_stochastic_behavior_model_rank_similarity_distance_based_minkowski_expanddims*
T0*/
_output_shapes
:??????????
div_no_nan_1DivNoNanNpow_1_stochastic_behavior_model_rank_similarity_distance_based_minkowski_pow_1Sdiv_no_nan_1_stochastic_behavior_model_rank_similarity_distance_based_minkowski_sum*
T0*/
_output_shapes
:?????????g
mul_5Multruediv_2:z:0div_no_nan_1:z:0*
T0*/
_output_shapes
:?????????L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
add_2AddV2Jpow_stochastic_behavior_model_rank_similarity_distance_based_minkowski_absadd_2/y:output:0*
T0*/
_output_shapes
:?????????O
LogLog	add_2:z:0*
T0*/
_output_shapes
:??????????
mul_6MulNmul_6_stochastic_behavior_model_rank_similarity_distance_based_minkowski_mul_1Log:y:0*
T0*/
_output_shapes
:?????????`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
SumSum	mul_6:z:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(_
mul_7Mul	mul_5:z:0Sum:output:0*
T0*/
_output_shapes
:?????????L
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
pow_3PowQsub_stochastic_behavior_model_rank_similarity_distance_based_minkowski_expanddimspow_3/y:output:0*
T0*/
_output_shapes
:?????????P
truediv_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??o
	truediv_3RealDivtruediv_3/x:output:0	pow_3:z:0*
T0*/
_output_shapes
:??????????
mul_8Multruediv_3:z:0Npow_1_stochastic_behavior_model_rank_similarity_distance_based_minkowski_pow_1*
T0*/
_output_shapes
:?????????L
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
add_3AddV2Sdiv_no_nan_1_stochastic_behavior_model_rank_similarity_distance_based_minkowski_sumadd_3/y:output:0*
T0*/
_output_shapes
:?????????Q
Log_1Log	add_3:z:0*
T0*/
_output_shapes
:?????????\
mul_9Mul	mul_8:z:0	Log_1:y:0*
T0*/
_output_shapes
:?????????\
sub_2Sub	mul_7:z:0	mul_9:z:0*
T0*/
_output_shapes
:?????????b
mul_10Mulresult_grads_0	sub_2:z:0*
T0*/
_output_shapes
:?????????t
SqueezeSqueeze
mul_10:z:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????[

Identity_1Identity	mul_4:z:0*
T0*/
_output_shapes
:?????????^

Identity_2IdentitySqueeze:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:_ [
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:_[
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_2:[W
+
_output_shapes
:?????????
(
_user_specified_nameresult_grads_3:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:5	1
/
_output_shapes
:?????????:5
1
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????
?
?
$__inference_signature_wrapper_118476
rank1_stimulus_set
agent_id
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5: 
	unknown_6
	unknown_7: 
	unknown_8:
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallrank1_stimulus_setagent_idunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*,
_read_only_resource_inputs


*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_117444s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:?????????:?????????:: : : : : : : : : : : : ::: 22
StatefulPartitionedCallStatefulPartitionedCall:` \
+
_output_shapes
:?????????
-
_user_specified_name2rank1/stimulus_set:UQ
+
_output_shapes
:?????????
"
_user_specified_name
agent_id: 

_output_shapes
::

_output_shapes
: :	

_output_shapes
: :$ 

_output_shapes

::,(
&
_output_shapes
::

_output_shapes
: 
?4
?
#__inference_internal_grad_fn_119994
result_grads_0
result_grads_1
result_grads_2
result_grads_3,
(mul_distance_based_minkowski_broadcastto$
 mul_distance_based_minkowski_sub$
 pow_distance_based_minkowski_abs+
'div_no_nan_distance_based_minkowski_pow+
'sub_distance_based_minkowski_expanddims(
$pow_1_distance_based_minkowski_pow_1-
)div_no_nan_1_distance_based_minkowski_sum(
$mul_6_distance_based_minkowski_mul_1
identity

identity_1

identity_2?
mulMul(mul_distance_based_minkowski_broadcastto mul_distance_based_minkowski_sub*
T0*/
_output_shapes
:?????????J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @v
powPow pow_distance_based_minkowski_abspow/y:output:0*
T0*/
_output_shapes
:??????????

div_no_nanDivNoNan'div_no_nan_distance_based_minkowski_powpow:z:0*
T0*/
_output_shapes
:?????????_
mul_1Mulmul:z:0div_no_nan:z:0*
T0*/
_output_shapes
:?????????J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??}
subSub'sub_distance_based_minkowski_expanddimssub/y:output:0*
T0*/
_output_shapes
:?????????u
pow_1Pow$pow_1_distance_based_minkowski_pow_1sub:z:0*
T0*/
_output_shapes
:?????????J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3a
addAddV2	pow_1:z:0add/y:output:0*
T0*/
_output_shapes
:?????????`
truedivRealDiv	mul_1:z:0add:z:0*
T0*/
_output_shapes
:?????????c
mul_2Mulresult_grads_0truediv:z:0*
T0*/
_output_shapes
:?????????L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
sub_1Sub'sub_distance_based_minkowski_expanddimssub_1/y:output:0*
T0*/
_output_shapes
:?????????w
pow_2Pow$pow_1_distance_based_minkowski_pow_1	sub_1:z:0*
T0*/
_output_shapes
:?????????z
mul_3Mul'sub_distance_based_minkowski_expanddims	pow_2:z:0*
T0*/
_output_shapes
:?????????L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3e
add_1AddV2	mul_3:z:0add_1/y:output:0*
T0*/
_output_shapes
:??????????
	truediv_1RealDiv'div_no_nan_distance_based_minkowski_pow	add_1:z:0*
T0*/
_output_shapes
:?????????e
mul_4Mulresult_grads_0truediv_1:z:0*
T0*/
_output_shapes
:?????????P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
	truediv_2RealDivtruediv_2/x:output:0'sub_distance_based_minkowski_expanddims*
T0*/
_output_shapes
:??????????
div_no_nan_1DivNoNan$pow_1_distance_based_minkowski_pow_1)div_no_nan_1_distance_based_minkowski_sum*
T0*/
_output_shapes
:?????????g
mul_5Multruediv_2:z:0div_no_nan_1:z:0*
T0*/
_output_shapes
:?????????L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3|
add_2AddV2 pow_distance_based_minkowski_absadd_2/y:output:0*
T0*/
_output_shapes
:?????????O
LogLog	add_2:z:0*
T0*/
_output_shapes
:?????????u
mul_6Mul$mul_6_distance_based_minkowski_mul_1Log:y:0*
T0*/
_output_shapes
:?????????`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
SumSum	mul_6:z:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(_
mul_7Mul	mul_5:z:0Sum:output:0*
T0*/
_output_shapes
:?????????L
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
pow_3Pow'sub_distance_based_minkowski_expanddimspow_3/y:output:0*
T0*/
_output_shapes
:?????????P
truediv_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??o
	truediv_3RealDivtruediv_3/x:output:0	pow_3:z:0*
T0*/
_output_shapes
:?????????{
mul_8Multruediv_3:z:0$pow_1_distance_based_minkowski_pow_1*
T0*/
_output_shapes
:?????????L
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
add_3AddV2)div_no_nan_1_distance_based_minkowski_sumadd_3/y:output:0*
T0*/
_output_shapes
:?????????Q
Log_1Log	add_3:z:0*
T0*/
_output_shapes
:?????????\
mul_9Mul	mul_8:z:0	Log_1:y:0*
T0*/
_output_shapes
:?????????\
sub_2Sub	mul_7:z:0	mul_9:z:0*
T0*/
_output_shapes
:?????????b
mul_10Mulresult_grads_0	sub_2:z:0*
T0*/
_output_shapes
:?????????t
SqueezeSqueeze
mul_10:z:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????[

Identity_1Identity	mul_4:z:0*
T0*/
_output_shapes
:?????????^

Identity_2IdentitySqueeze:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:_ [
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:_[
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_2:[W
+
_output_shapes
:?????????
(
_user_specified_nameresult_grads_3:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:5	1
/
_output_shapes
:?????????:5
1
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????
?
?
:__inference_stochastic_behavior_model_layer_call_fn_118515
inputs_2rank1_stimulus_set
inputs_agent_id
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5: 
	unknown_6
	unknown_7: 
	unknown_8:
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_2rank1_stimulus_setinputs_agent_idunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:?????????: *,
_read_only_resource_inputs


*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_117784s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:?????????:?????????:: : : : : : : : : : : : ::: 22
StatefulPartitionedCallStatefulPartitionedCall:g c
+
_output_shapes
:?????????
4
_user_specified_nameinputs/2rank1/stimulus_set:\X
+
_output_shapes
:?????????
)
_user_specified_nameinputs/agent_id: 

_output_shapes
::

_output_shapes
: :	

_output_shapes
: :$ 

_output_shapes

::,(
&
_output_shapes
::

_output_shapes
: 
?7
?
#__inference_internal_grad_fn_120238
result_grads_0
result_grads_1
result_grads_2
result_grads_3<
8mul_rank_similarity_distance_based_minkowski_broadcastto4
0mul_rank_similarity_distance_based_minkowski_sub4
0pow_rank_similarity_distance_based_minkowski_abs;
7div_no_nan_rank_similarity_distance_based_minkowski_pow;
7sub_rank_similarity_distance_based_minkowski_expanddims8
4pow_1_rank_similarity_distance_based_minkowski_pow_1=
9div_no_nan_1_rank_similarity_distance_based_minkowski_sum8
4mul_6_rank_similarity_distance_based_minkowski_mul_1
identity

identity_1

identity_2?
mulMul8mul_rank_similarity_distance_based_minkowski_broadcastto0mul_rank_similarity_distance_based_minkowski_sub*
T0*/
_output_shapes
:?????????J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
powPow0pow_rank_similarity_distance_based_minkowski_abspow/y:output:0*
T0*/
_output_shapes
:??????????

div_no_nanDivNoNan7div_no_nan_rank_similarity_distance_based_minkowski_powpow:z:0*
T0*/
_output_shapes
:?????????_
mul_1Mulmul:z:0div_no_nan:z:0*
T0*/
_output_shapes
:?????????J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
subSub7sub_rank_similarity_distance_based_minkowski_expanddimssub/y:output:0*
T0*/
_output_shapes
:??????????
pow_1Pow4pow_1_rank_similarity_distance_based_minkowski_pow_1sub:z:0*
T0*/
_output_shapes
:?????????J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3a
addAddV2	pow_1:z:0add/y:output:0*
T0*/
_output_shapes
:?????????`
truedivRealDiv	mul_1:z:0add:z:0*
T0*/
_output_shapes
:?????????c
mul_2Mulresult_grads_0truediv:z:0*
T0*/
_output_shapes
:?????????L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
sub_1Sub7sub_rank_similarity_distance_based_minkowski_expanddimssub_1/y:output:0*
T0*/
_output_shapes
:??????????
pow_2Pow4pow_1_rank_similarity_distance_based_minkowski_pow_1	sub_1:z:0*
T0*/
_output_shapes
:??????????
mul_3Mul7sub_rank_similarity_distance_based_minkowski_expanddims	pow_2:z:0*
T0*/
_output_shapes
:?????????L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3e
add_1AddV2	mul_3:z:0add_1/y:output:0*
T0*/
_output_shapes
:??????????
	truediv_1RealDiv7div_no_nan_rank_similarity_distance_based_minkowski_pow	add_1:z:0*
T0*/
_output_shapes
:?????????e
mul_4Mulresult_grads_0truediv_1:z:0*
T0*/
_output_shapes
:?????????P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
	truediv_2RealDivtruediv_2/x:output:07sub_rank_similarity_distance_based_minkowski_expanddims*
T0*/
_output_shapes
:??????????
div_no_nan_1DivNoNan4pow_1_rank_similarity_distance_based_minkowski_pow_19div_no_nan_1_rank_similarity_distance_based_minkowski_sum*
T0*/
_output_shapes
:?????????g
mul_5Multruediv_2:z:0div_no_nan_1:z:0*
T0*/
_output_shapes
:?????????L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
add_2AddV20pow_rank_similarity_distance_based_minkowski_absadd_2/y:output:0*
T0*/
_output_shapes
:?????????O
LogLog	add_2:z:0*
T0*/
_output_shapes
:??????????
mul_6Mul4mul_6_rank_similarity_distance_based_minkowski_mul_1Log:y:0*
T0*/
_output_shapes
:?????????`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
SumSum	mul_6:z:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(_
mul_7Mul	mul_5:z:0Sum:output:0*
T0*/
_output_shapes
:?????????L
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
pow_3Pow7sub_rank_similarity_distance_based_minkowski_expanddimspow_3/y:output:0*
T0*/
_output_shapes
:?????????P
truediv_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??o
	truediv_3RealDivtruediv_3/x:output:0	pow_3:z:0*
T0*/
_output_shapes
:??????????
mul_8Multruediv_3:z:04pow_1_rank_similarity_distance_based_minkowski_pow_1*
T0*/
_output_shapes
:?????????L
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
add_3AddV29div_no_nan_1_rank_similarity_distance_based_minkowski_sumadd_3/y:output:0*
T0*/
_output_shapes
:?????????Q
Log_1Log	add_3:z:0*
T0*/
_output_shapes
:?????????\
mul_9Mul	mul_8:z:0	Log_1:y:0*
T0*/
_output_shapes
:?????????\
sub_2Sub	mul_7:z:0	mul_9:z:0*
T0*/
_output_shapes
:?????????b
mul_10Mulresult_grads_0	sub_2:z:0*
T0*/
_output_shapes
:?????????t
SqueezeSqueeze
mul_10:z:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????[

Identity_1Identity	mul_4:z:0*
T0*/
_output_shapes
:?????????^

Identity_2IdentitySqueeze:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:_ [
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:_[
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_2:[W
+
_output_shapes
:?????????
(
_user_specified_nameresult_grads_3:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:5	1
/
_output_shapes
:?????????:5
1
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????
?
?
:__inference_stochastic_behavior_model_layer_call_fn_117820
rank1_stimulus_set
agent_id
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5: 
	unknown_6
	unknown_7: 
	unknown_8:
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallrank1_stimulus_setagent_idunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:?????????: *,
_read_only_resource_inputs


*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_117784s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:?????????:?????????:: : : : : : : : : : : : ::: 22
StatefulPartitionedCallStatefulPartitionedCall:` \
+
_output_shapes
:?????????
-
_user_specified_name2rank1/stimulus_set:UQ
+
_output_shapes
:?????????
"
_user_specified_name
agent_id: 

_output_shapes
::

_output_shapes
: :	

_output_shapes
: :$ 

_output_shapes

::,(
&
_output_shapes
::

_output_shapes
: 
?4
?
#__inference_internal_grad_fn_119933
result_grads_0
result_grads_1
result_grads_2
result_grads_3,
(mul_distance_based_minkowski_broadcastto$
 mul_distance_based_minkowski_sub$
 pow_distance_based_minkowski_abs+
'div_no_nan_distance_based_minkowski_pow+
'sub_distance_based_minkowski_expanddims(
$pow_1_distance_based_minkowski_pow_1-
)div_no_nan_1_distance_based_minkowski_sum(
$mul_6_distance_based_minkowski_mul_1
identity

identity_1

identity_2?
mulMul(mul_distance_based_minkowski_broadcastto mul_distance_based_minkowski_sub*
T0*/
_output_shapes
:?????????J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @v
powPow pow_distance_based_minkowski_abspow/y:output:0*
T0*/
_output_shapes
:??????????

div_no_nanDivNoNan'div_no_nan_distance_based_minkowski_powpow:z:0*
T0*/
_output_shapes
:?????????_
mul_1Mulmul:z:0div_no_nan:z:0*
T0*/
_output_shapes
:?????????J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??}
subSub'sub_distance_based_minkowski_expanddimssub/y:output:0*
T0*/
_output_shapes
:?????????u
pow_1Pow$pow_1_distance_based_minkowski_pow_1sub:z:0*
T0*/
_output_shapes
:?????????J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3a
addAddV2	pow_1:z:0add/y:output:0*
T0*/
_output_shapes
:?????????`
truedivRealDiv	mul_1:z:0add:z:0*
T0*/
_output_shapes
:?????????c
mul_2Mulresult_grads_0truediv:z:0*
T0*/
_output_shapes
:?????????L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
sub_1Sub'sub_distance_based_minkowski_expanddimssub_1/y:output:0*
T0*/
_output_shapes
:?????????w
pow_2Pow$pow_1_distance_based_minkowski_pow_1	sub_1:z:0*
T0*/
_output_shapes
:?????????z
mul_3Mul'sub_distance_based_minkowski_expanddims	pow_2:z:0*
T0*/
_output_shapes
:?????????L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3e
add_1AddV2	mul_3:z:0add_1/y:output:0*
T0*/
_output_shapes
:??????????
	truediv_1RealDiv'div_no_nan_distance_based_minkowski_pow	add_1:z:0*
T0*/
_output_shapes
:?????????e
mul_4Mulresult_grads_0truediv_1:z:0*
T0*/
_output_shapes
:?????????P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
	truediv_2RealDivtruediv_2/x:output:0'sub_distance_based_minkowski_expanddims*
T0*/
_output_shapes
:??????????
div_no_nan_1DivNoNan$pow_1_distance_based_minkowski_pow_1)div_no_nan_1_distance_based_minkowski_sum*
T0*/
_output_shapes
:?????????g
mul_5Multruediv_2:z:0div_no_nan_1:z:0*
T0*/
_output_shapes
:?????????L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3|
add_2AddV2 pow_distance_based_minkowski_absadd_2/y:output:0*
T0*/
_output_shapes
:?????????O
LogLog	add_2:z:0*
T0*/
_output_shapes
:?????????u
mul_6Mul$mul_6_distance_based_minkowski_mul_1Log:y:0*
T0*/
_output_shapes
:?????????`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
SumSum	mul_6:z:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(_
mul_7Mul	mul_5:z:0Sum:output:0*
T0*/
_output_shapes
:?????????L
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
pow_3Pow'sub_distance_based_minkowski_expanddimspow_3/y:output:0*
T0*/
_output_shapes
:?????????P
truediv_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??o
	truediv_3RealDivtruediv_3/x:output:0	pow_3:z:0*
T0*/
_output_shapes
:?????????{
mul_8Multruediv_3:z:0$pow_1_distance_based_minkowski_pow_1*
T0*/
_output_shapes
:?????????L
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
add_3AddV2)div_no_nan_1_distance_based_minkowski_sumadd_3/y:output:0*
T0*/
_output_shapes
:?????????Q
Log_1Log	add_3:z:0*
T0*/
_output_shapes
:?????????\
mul_9Mul	mul_8:z:0	Log_1:y:0*
T0*/
_output_shapes
:?????????\
sub_2Sub	mul_7:z:0	mul_9:z:0*
T0*/
_output_shapes
:?????????b
mul_10Mulresult_grads_0	sub_2:z:0*
T0*/
_output_shapes
:?????????t
SqueezeSqueeze
mul_10:z:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????[

Identity_1Identity	mul_4:z:0*
T0*/
_output_shapes
:?????????^

Identity_2IdentitySqueeze:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:_ [
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:_[
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_2:[W
+
_output_shapes
:?????????
(
_user_specified_nameresult_grads_3:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:5	1
/
_output_shapes
:?????????:5
1
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????
?
?
U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_117784

inputs
inputs_1
rank_similarity_117748
rank_similarity_117750(
rank_similarity_117752:(
rank_similarity_117754:(
rank_similarity_117756:(
rank_similarity_117758: 
rank_similarity_117760: 
rank_similarity_117762 
rank_similarity_117764: $
rank_similarity_117766: 
rank_similarity_117768:  
rank_similarity_117770:  
rank_similarity_117772: 
rank_similarity_117774
rank_similarity_117776
rank_similarity_117778
identity

identity_1??'rank_similarity/StatefulPartitionedCall?
'rank_similarity/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1rank_similarity_117748rank_similarity_117750rank_similarity_117752rank_similarity_117754rank_similarity_117756rank_similarity_117758rank_similarity_117760rank_similarity_117762rank_similarity_117764rank_similarity_117766rank_similarity_117768rank_similarity_117770rank_similarity_117772rank_similarity_117774rank_similarity_117776rank_similarity_117778*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:?????????: *,
_read_only_resource_inputs


*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_rank_similarity_layer_call_and_return_conditional_losses_117747?
IdentityIdentity0rank_similarity/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????p

Identity_1Identity0rank_similarity/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: p
NoOpNoOp(^rank_similarity/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:?????????:?????????:: : : : : : : : : : : : ::: 2R
'rank_similarity/StatefulPartitionedCall'rank_similarity/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????
 
_user_specified_nameinputs: 

_output_shapes
::

_output_shapes
: :	

_output_shapes
: :$ 

_output_shapes

::,(
&
_output_shapes
::

_output_shapes
: 
?9
?
__inference__traced_save_120344
file_prefix8
4savev2_embedding_normal_diag_loc_read_readvariableopH
Dsavev2_embedding_normal_diag_untransformed_scale_read_readvariableopJ
Fsavev2_embedding_normal_diag_1_untransformed_scale_read_readvariableop(
$savev2_kl_anneal_read_readvariableop:
6savev2_embedding_normal_diag_1_loc_read_readvariableop,
(savev2_minkowski_rho_read_readvariableopc
_savev2_stochastic_behavior_model_rank_similarity_distance_based_minkowski_w_read_readvariableop9
5savev2_exponential_similarity_tau_read_readvariableop;
7savev2_exponential_similarity_gamma_read_readvariableop:
6savev2_exponential_similarity_beta_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop?
;savev2_adam_embedding_normal_diag_loc_m_read_readvariableopO
Ksavev2_adam_embedding_normal_diag_untransformed_scale_m_read_readvariableopQ
Msavev2_adam_embedding_normal_diag_1_untransformed_scale_m_read_readvariableop?
;savev2_adam_embedding_normal_diag_loc_v_read_readvariableopO
Ksavev2_adam_embedding_normal_diag_untransformed_scale_v_read_readvariableopQ
Msavev2_adam_embedding_normal_diag_1_untransformed_scale_v_read_readvariableop
savev2_const_6

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?

value?
B?
B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_embedding_normal_diag_loc_read_readvariableopDsavev2_embedding_normal_diag_untransformed_scale_read_readvariableopFsavev2_embedding_normal_diag_1_untransformed_scale_read_readvariableop$savev2_kl_anneal_read_readvariableop6savev2_embedding_normal_diag_1_loc_read_readvariableop(savev2_minkowski_rho_read_readvariableop_savev2_stochastic_behavior_model_rank_similarity_distance_based_minkowski_w_read_readvariableop5savev2_exponential_similarity_tau_read_readvariableop7savev2_exponential_similarity_gamma_read_readvariableop6savev2_exponential_similarity_beta_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop;savev2_adam_embedding_normal_diag_loc_m_read_readvariableopKsavev2_adam_embedding_normal_diag_untransformed_scale_m_read_readvariableopMsavev2_adam_embedding_normal_diag_1_untransformed_scale_m_read_readvariableop;savev2_adam_embedding_normal_diag_loc_v_read_readvariableopKsavev2_adam_embedding_normal_diag_untransformed_scale_v_read_readvariableopMsavev2_adam_embedding_normal_diag_1_untransformed_scale_v_read_readvariableopsavev2_const_6"/device:CPU:0*
_output_shapes
 *(
dtypes
2	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :::: :: :: : : : : : : : : : : : ::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: : 

_output_shapes
::

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::

_output_shapes
: 
?4
?
#__inference_internal_grad_fn_120055
result_grads_0
result_grads_1
result_grads_2
result_grads_3,
(mul_distance_based_minkowski_broadcastto$
 mul_distance_based_minkowski_sub$
 pow_distance_based_minkowski_abs+
'div_no_nan_distance_based_minkowski_pow+
'sub_distance_based_minkowski_expanddims(
$pow_1_distance_based_minkowski_pow_1-
)div_no_nan_1_distance_based_minkowski_sum(
$mul_6_distance_based_minkowski_mul_1
identity

identity_1

identity_2?
mulMul(mul_distance_based_minkowski_broadcastto mul_distance_based_minkowski_sub*
T0*/
_output_shapes
:?????????J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @v
powPow pow_distance_based_minkowski_abspow/y:output:0*
T0*/
_output_shapes
:??????????

div_no_nanDivNoNan'div_no_nan_distance_based_minkowski_powpow:z:0*
T0*/
_output_shapes
:?????????_
mul_1Mulmul:z:0div_no_nan:z:0*
T0*/
_output_shapes
:?????????J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??}
subSub'sub_distance_based_minkowski_expanddimssub/y:output:0*
T0*/
_output_shapes
:?????????u
pow_1Pow$pow_1_distance_based_minkowski_pow_1sub:z:0*
T0*/
_output_shapes
:?????????J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3a
addAddV2	pow_1:z:0add/y:output:0*
T0*/
_output_shapes
:?????????`
truedivRealDiv	mul_1:z:0add:z:0*
T0*/
_output_shapes
:?????????c
mul_2Mulresult_grads_0truediv:z:0*
T0*/
_output_shapes
:?????????L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
sub_1Sub'sub_distance_based_minkowski_expanddimssub_1/y:output:0*
T0*/
_output_shapes
:?????????w
pow_2Pow$pow_1_distance_based_minkowski_pow_1	sub_1:z:0*
T0*/
_output_shapes
:?????????z
mul_3Mul'sub_distance_based_minkowski_expanddims	pow_2:z:0*
T0*/
_output_shapes
:?????????L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3e
add_1AddV2	mul_3:z:0add_1/y:output:0*
T0*/
_output_shapes
:??????????
	truediv_1RealDiv'div_no_nan_distance_based_minkowski_pow	add_1:z:0*
T0*/
_output_shapes
:?????????e
mul_4Mulresult_grads_0truediv_1:z:0*
T0*/
_output_shapes
:?????????P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
	truediv_2RealDivtruediv_2/x:output:0'sub_distance_based_minkowski_expanddims*
T0*/
_output_shapes
:??????????
div_no_nan_1DivNoNan$pow_1_distance_based_minkowski_pow_1)div_no_nan_1_distance_based_minkowski_sum*
T0*/
_output_shapes
:?????????g
mul_5Multruediv_2:z:0div_no_nan_1:z:0*
T0*/
_output_shapes
:?????????L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3|
add_2AddV2 pow_distance_based_minkowski_absadd_2/y:output:0*
T0*/
_output_shapes
:?????????O
LogLog	add_2:z:0*
T0*/
_output_shapes
:?????????u
mul_6Mul$mul_6_distance_based_minkowski_mul_1Log:y:0*
T0*/
_output_shapes
:?????????`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
SumSum	mul_6:z:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(_
mul_7Mul	mul_5:z:0Sum:output:0*
T0*/
_output_shapes
:?????????L
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
pow_3Pow'sub_distance_based_minkowski_expanddimspow_3/y:output:0*
T0*/
_output_shapes
:?????????P
truediv_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??o
	truediv_3RealDivtruediv_3/x:output:0	pow_3:z:0*
T0*/
_output_shapes
:?????????{
mul_8Multruediv_3:z:0$pow_1_distance_based_minkowski_pow_1*
T0*/
_output_shapes
:?????????L
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
add_3AddV2)div_no_nan_1_distance_based_minkowski_sumadd_3/y:output:0*
T0*/
_output_shapes
:?????????Q
Log_1Log	add_3:z:0*
T0*/
_output_shapes
:?????????\
mul_9Mul	mul_8:z:0	Log_1:y:0*
T0*/
_output_shapes
:?????????\
sub_2Sub	mul_7:z:0	mul_9:z:0*
T0*/
_output_shapes
:?????????b
mul_10Mulresult_grads_0	sub_2:z:0*
T0*/
_output_shapes
:?????????t
SqueezeSqueeze
mul_10:z:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????[

Identity_1Identity	mul_4:z:0*
T0*/
_output_shapes
:?????????^

Identity_2IdentitySqueeze:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:_ [
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:_[
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_2:[W
+
_output_shapes
:?????????
(
_user_specified_nameresult_grads_3:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:5	1
/
_output_shapes
:?????????:5
1
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????
?4
?
#__inference_internal_grad_fn_120116
result_grads_0
result_grads_1
result_grads_2
result_grads_3,
(mul_distance_based_minkowski_broadcastto$
 mul_distance_based_minkowski_sub$
 pow_distance_based_minkowski_abs+
'div_no_nan_distance_based_minkowski_pow+
'sub_distance_based_minkowski_expanddims(
$pow_1_distance_based_minkowski_pow_1-
)div_no_nan_1_distance_based_minkowski_sum(
$mul_6_distance_based_minkowski_mul_1
identity

identity_1

identity_2?
mulMul(mul_distance_based_minkowski_broadcastto mul_distance_based_minkowski_sub*
T0*/
_output_shapes
:?????????J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @v
powPow pow_distance_based_minkowski_abspow/y:output:0*
T0*/
_output_shapes
:??????????

div_no_nanDivNoNan'div_no_nan_distance_based_minkowski_powpow:z:0*
T0*/
_output_shapes
:?????????_
mul_1Mulmul:z:0div_no_nan:z:0*
T0*/
_output_shapes
:?????????J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??}
subSub'sub_distance_based_minkowski_expanddimssub/y:output:0*
T0*/
_output_shapes
:?????????u
pow_1Pow$pow_1_distance_based_minkowski_pow_1sub:z:0*
T0*/
_output_shapes
:?????????J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3a
addAddV2	pow_1:z:0add/y:output:0*
T0*/
_output_shapes
:?????????`
truedivRealDiv	mul_1:z:0add:z:0*
T0*/
_output_shapes
:?????????c
mul_2Mulresult_grads_0truediv:z:0*
T0*/
_output_shapes
:?????????L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
sub_1Sub'sub_distance_based_minkowski_expanddimssub_1/y:output:0*
T0*/
_output_shapes
:?????????w
pow_2Pow$pow_1_distance_based_minkowski_pow_1	sub_1:z:0*
T0*/
_output_shapes
:?????????z
mul_3Mul'sub_distance_based_minkowski_expanddims	pow_2:z:0*
T0*/
_output_shapes
:?????????L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3e
add_1AddV2	mul_3:z:0add_1/y:output:0*
T0*/
_output_shapes
:??????????
	truediv_1RealDiv'div_no_nan_distance_based_minkowski_pow	add_1:z:0*
T0*/
_output_shapes
:?????????e
mul_4Mulresult_grads_0truediv_1:z:0*
T0*/
_output_shapes
:?????????P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
	truediv_2RealDivtruediv_2/x:output:0'sub_distance_based_minkowski_expanddims*
T0*/
_output_shapes
:??????????
div_no_nan_1DivNoNan$pow_1_distance_based_minkowski_pow_1)div_no_nan_1_distance_based_minkowski_sum*
T0*/
_output_shapes
:?????????g
mul_5Multruediv_2:z:0div_no_nan_1:z:0*
T0*/
_output_shapes
:?????????L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3|
add_2AddV2 pow_distance_based_minkowski_absadd_2/y:output:0*
T0*/
_output_shapes
:?????????O
LogLog	add_2:z:0*
T0*/
_output_shapes
:?????????u
mul_6Mul$mul_6_distance_based_minkowski_mul_1Log:y:0*
T0*/
_output_shapes
:?????????`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
SumSum	mul_6:z:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(_
mul_7Mul	mul_5:z:0Sum:output:0*
T0*/
_output_shapes
:?????????L
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
pow_3Pow'sub_distance_based_minkowski_expanddimspow_3/y:output:0*
T0*/
_output_shapes
:?????????P
truediv_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??o
	truediv_3RealDivtruediv_3/x:output:0	pow_3:z:0*
T0*/
_output_shapes
:?????????{
mul_8Multruediv_3:z:0$pow_1_distance_based_minkowski_pow_1*
T0*/
_output_shapes
:?????????L
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
add_3AddV2)div_no_nan_1_distance_based_minkowski_sumadd_3/y:output:0*
T0*/
_output_shapes
:?????????Q
Log_1Log	add_3:z:0*
T0*/
_output_shapes
:?????????\
mul_9Mul	mul_8:z:0	Log_1:y:0*
T0*/
_output_shapes
:?????????\
sub_2Sub	mul_7:z:0	mul_9:z:0*
T0*/
_output_shapes
:?????????b
mul_10Mulresult_grads_0	sub_2:z:0*
T0*/
_output_shapes
:?????????t
SqueezeSqueeze
mul_10:z:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????[

Identity_1Identity	mul_4:z:0*
T0*/
_output_shapes
:?????????^

Identity_2IdentitySqueeze:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:_ [
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:_[
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_2:[W
+
_output_shapes
:?????????
(
_user_specified_nameresult_grads_3:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:5	1
/
_output_shapes
:?????????:5
1
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????
?
?
U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_118275

inputs
inputs_1
rank_similarity_118239
rank_similarity_118241(
rank_similarity_118243:(
rank_similarity_118245:(
rank_similarity_118247:(
rank_similarity_118249: 
rank_similarity_118251: 
rank_similarity_118253 
rank_similarity_118255: $
rank_similarity_118257: 
rank_similarity_118259:  
rank_similarity_118261:  
rank_similarity_118263: 
rank_similarity_118265
rank_similarity_118267
rank_similarity_118269
identity

identity_1??'rank_similarity/StatefulPartitionedCall?
'rank_similarity/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1rank_similarity_118239rank_similarity_118241rank_similarity_118243rank_similarity_118245rank_similarity_118247rank_similarity_118249rank_similarity_118251rank_similarity_118253rank_similarity_118255rank_similarity_118257rank_similarity_118259rank_similarity_118261rank_similarity_118263rank_similarity_118265rank_similarity_118267rank_similarity_118269*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:?????????: *,
_read_only_resource_inputs


*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_rank_similarity_layer_call_and_return_conditional_losses_118157?
IdentityIdentity0rank_similarity/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????p

Identity_1Identity0rank_similarity/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: p
NoOpNoOp(^rank_similarity/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:?????????:?????????:: : : : : : : : : : : : ::: 2R
'rank_similarity/StatefulPartitionedCall'rank_similarity/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????
 
_user_specified_nameinputs: 

_output_shapes
::

_output_shapes
: :	

_output_shapes
: :$ 

_output_shapes

::,(
&
_output_shapes
::

_output_shapes
: 
??
?
U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_118848
inputs_2rank1_stimulus_set
inputs_agent_id$
 rank_similarity_gatherv2_indices!
rank_similarity_gatherv2_axise
Srank_similarity_embedding_variational_embedding_normal_diag_embedding_lookup_118563:?
orank_similarity_embedding_variational_embedding_normal_diag_embedding_lookup_1_softplus_readvariableop_resource:x
frank_similarity_embedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_118612:?
?rank_similarity_embedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_1_softplus_readvariableop_resource:M
Crank_similarity_embedding_variational_mul_1_readvariableop_resource: #
rank_similarity_packed_values_1J
@rank_similarity_distance_based_minkowski_readvariableop_resource: Z
Lrank_similarity_distance_based_minkowski_broadcastto_readvariableop_resource:[
Qrank_similarity_distance_based_exponential_similarity_neg_readvariableop_resource: [
Qrank_similarity_distance_based_exponential_similarity_pow_readvariableop_resource: [
Qrank_similarity_distance_based_exponential_similarity_add_readvariableop_resource: &
"rank_similarity_gatherv2_1_indices
rank_similarity_mul_x
rank_similarity_truediv_y
identity

identity_1??Hrank_similarity/distance_based/exponential_similarity/Neg/ReadVariableOp?Hrank_similarity/distance_based/exponential_similarity/Pow/ReadVariableOp?Hrank_similarity/distance_based/exponential_similarity/add/ReadVariableOp?Crank_similarity/distance_based/minkowski/BroadcastTo/ReadVariableOp?7rank_similarity/distance_based/minkowski/ReadVariableOp??rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOp??rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOp??rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOp??rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOp?Lrank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup?frank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOp?_rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup?yrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOp?:rank_similarity/embedding_variational/mul_1/ReadVariableOp??rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp??rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp?
rank_similarity/GatherV2GatherV2inputs_2rank1_stimulus_set rank_similarity_gatherv2_indicesrank_similarity_gatherv2_axis*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:?????????\
rank_similarity/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
rank_similarity/NotEqualNotEqual!rank_similarity/GatherV2:output:0#rank_similarity/NotEqual/y:output:0*
T0*+
_output_shapes
:??????????
Lrank_similarity/embedding_variational/embedding_normal_diag/embedding_lookupResourceGatherSrank_similarity_embedding_variational_embedding_normal_diag_embedding_lookup_118563inputs_2rank1_stimulus_set*
Tindices0*f
_class\
ZXloc:@rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup/118563*/
_output_shapes
:?????????*
dtype0?
Urank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup/IdentityIdentityUrank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup:output:0*
T0*f
_class\
ZXloc:@rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup/118563*/
_output_shapes
:??????????
Wrank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup/Identity_1Identity^rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup/Identity:output:0*
T0*/
_output_shapes
:??????????
frank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOpReadVariableOporank_similarity_embedding_variational_embedding_normal_diag_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
Wrank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/SoftplusSoftplusnrank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
Trank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
Rrank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/addAddV2]rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/add/x:output:0erank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus:activations:0*
T0*
_output_shapes

:?
Srank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/axisConst*e
_class[
YWloc:@rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/add*
_output_shapes
: *
dtype0*
value	B : ?
Nrank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1GatherV2Vrank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/add:z:0inputs_2rank1_stimulus_set\rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*e
_class[
YWloc:@rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/add*/
_output_shapes
:??????????
Wrank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/IdentityIdentityWrank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1:output:0*
T0*/
_output_shapes
:??????????
Vrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Orank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/ShapeShape`rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
Orank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
]rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
_rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
_rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Wrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_sliceStridedSliceXrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/Shape:output:0frank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack:output:0hrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack_1:output:0hrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Qrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/Shape_1Shape`rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/Identity:output:0*
T0*
_output_shapes
:?
Qrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : ?
_rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
arank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
arank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Yrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1StridedSliceZrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/Shape_1:output:0hrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack:output:0jrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack_1:output:0jrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Zrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
\rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
Wrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgsBroadcastArgserank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs/s0_1:output:0`rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice:output:0*
_output_shapes
:?
Yrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs_1BroadcastArgs\rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs:r0:0brank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1:output:0*
_output_shapes
:?
Yrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
Urank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Prank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/concatConcatV2brank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/concat/values_0:output:0^rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs_1:r0:0^rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
crank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
erank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
srank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalYrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/concat:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????*
dtype0?
brank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/mulMul|rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/RandomStandardNormal:output:0nrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/stddev:output:0*
T0*N
_output_shapes<
::8?????????????????????????????????????
^rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normalAddV2frank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/mul:z:0lrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/mean:output:0*
T0*N
_output_shapes<
::8?????????????????????????????????????
Mrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/mulMulbrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal:z:0`rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/Identity:output:0*
T0*E
_output_shapes3
1:/????????????????????????????
Mrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/addAddV2Qrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/mul:z:0`rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup/Identity_1:output:0*
T0*E
_output_shapes3
1:/????????????????????????????
Qrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/Shape_2ShapeQrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/add:z:0*
T0*
_output_shapes
:?
_rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
arank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
arank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Yrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2StridedSliceZrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/Shape_2:output:0hrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack:output:0jrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack_1:output:0jrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Wrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Rrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/concat_1ConcatV2_rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/sample_shape:output:0brank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2:output:0`rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Qrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/ReshapeReshapeQrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/add:z:0[rank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/concat_1:output:0*
T0*/
_output_shapes
:??????????
Arank_similarity/embedding_variational/embedding_shared/zeros_like	ZerosLikeinputs_2rank1_stimulus_set*
T0*+
_output_shapes
:??????????
_rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookupResourceGatherfrank_similarity_embedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_118612Erank_similarity/embedding_variational/embedding_shared/zeros_like:y:0*
Tindices0*y
_classo
mkloc:@rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/118612*/
_output_shapes
:?????????*
dtype0?
hrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/IdentityIdentityhrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup:output:0*
T0*y
_classo
mkloc:@rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/118612*/
_output_shapes
:??????????
jrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/Identity_1Identityqrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/Identity:output:0*
T0*/
_output_shapes
:??????????
yrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOpReadVariableOp?rank_similarity_embedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
jrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/SoftplusSoftplus?rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
grank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
erank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/addAddV2prank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add/x:output:0xrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus:activations:0*
T0*
_output_shapes

:?
frank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/axisConst*x
_classn
ljloc:@rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add*
_output_shapes
: *
dtype0*
value	B : ?
arank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1GatherV2irank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add:z:0Erank_similarity/embedding_variational/embedding_shared/zeros_like:y:0orank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*x
_classn
ljloc:@rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add*/
_output_shapes
:??????????
jrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/IdentityIdentityjrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1:output:0*
T0*/
_output_shapes
:??????????
irank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
brank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/ShapeShapesrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
brank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
prank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
rrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
rrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
jrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_sliceStridedSlicekrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape:output:0yrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack:output:0{rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack_1:output:0{rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
drank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape_1Shapesrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Identity:output:0*
T0*
_output_shapes
:?
drank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : ?
rrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
trank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
trank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1StridedSlicemrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape_1:output:0{rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack:output:0}rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack_1:output:0}rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
mrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
orank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
jrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgsBroadcastArgsxrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs/s0_1:output:0srank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice:output:0*
_output_shapes
:?
lrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs_1BroadcastArgsorank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs:r0:0urank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1:output:0*
_output_shapes
:?
lrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
hrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
crank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concatConcatV2urank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat/values_0:output:0qrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs_1:r0:0qrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
vrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
xrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
?rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormallrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????*
dtype0?
urank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/mulMul?rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/RandomStandardNormal:output:0?rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/stddev:output:0*
T0*N
_output_shapes<
::8?????????????????????????????????????
qrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normalAddV2yrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/mul:z:0rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/mean:output:0*
T0*N
_output_shapes<
::8?????????????????????????????????????
`rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/mulMulurank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal:z:0srank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Identity:output:0*
T0*E
_output_shapes3
1:/????????????????????????????
`rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/addAddV2drank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/mul:z:0srank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/Identity_1:output:0*
T0*E
_output_shapes3
1:/????????????????????????????
drank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape_2Shapedrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/add:z:0*
T0*
_output_shapes
:?
rrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
trank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
trank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2StridedSlicemrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape_2:output:0{rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack:output:0}rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack_1:output:0}rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
jrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
erank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat_1ConcatV2rrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/sample_shape:output:0urank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2:output:0srank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
drank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/ReshapeReshapedrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/add:z:0nrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat_1:output:0*
T0*/
_output_shapes
:??????????
=rank_similarity/embedding_variational/reshape/event_shape_outConst*
_output_shapes
: *
dtype0*
valueB ?
<rank_similarity/embedding_variational/reshape/event_shape_inConst*
_output_shapes
:*
dtype0*
valueB"      ?
Jrank_similarity/embedding_variational/SampleIndependentNormal/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
Irank_similarity/embedding_variational/reshapeSampleIndependentNormal/zeroConst*
_output_shapes
: *
dtype0*
value	B : ?
prank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
~rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOpReadVariableOpSrank_similarity_embedding_variational_embedding_normal_diag_embedding_lookup_118563*
_output_shapes

:*
dtype0?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOpReadVariableOporank_similarity_embedding_variational_embedding_normal_diag_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
zrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/SoftplusSoftplus?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
wrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
urank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/addAddV2?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add/x:output:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus:activations:0*
T0*
_output_shapes

:?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"      ?
wrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_sliceStridedSlice?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/shape_as_tensor:output:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack:output:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack_1:output:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"      ?
yrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : ?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1StridedSlice?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/shape_as_tensor_1:output:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack:output:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack_1:output:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgsBroadcastArgs?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs/s0_1:output:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice:output:0*
_output_shapes
:?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs_1BroadcastArgs?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs:r0:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1:output:0*
_output_shapes
:?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
}rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
xrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concatConcatV2?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat/values_0:output:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs_1:r0:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat:output:0*
T0*"
_output_shapes
:*
dtype0?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/mulMul?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/RandomStandardNormal:output:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/stddev:output:0*
T0*"
_output_shapes
:?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normalAddV2?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/mul:z:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/mean:output:0*
T0*"
_output_shapes
:?
urank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/mulMul?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal:z:0yrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add:z:0*
T0*"
_output_shapes
:?
wrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add_1AddV2yrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/mul:z:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOp:value:0*
T0*"
_output_shapes
:?
rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
yrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReshapeReshape{rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add_1:z:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Reshape/shape:output:0*
T0*"
_output_shapes
:?
qrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
krank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/ReshapeReshape?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Reshape:output:0zrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Reshape/shape:output:0*
T0*"
_output_shapes
:?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOpReadVariableOporank_similarity_embedding_variational_embedding_normal_diag_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
~rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/SoftplusSoftplus?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
{rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
yrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/addAddV2?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add/x:output:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus:activations:0*
T0*
_output_shapes

:?
}rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/truedivRealDivtrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Reshape:output:0}rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add:z:0*
T0*"
_output_shapes
:?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOpReadVariableOpSrank_similarity_embedding_variational_embedding_normal_diag_embedding_lookup_118563*
_output_shapes

:*
dtype0?
rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/truediv_1RealDiv?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOp:value:0}rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add:z:0*
T0*
_output_shapes

:?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/SquaredDifferenceSquaredDifference?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/truediv:z:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/truediv_1:z:0*
T0*"
_output_shapes
:?
{rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
yrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/mulMul?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/mul/x:output:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/SquaredDifference:z:0*
T0*"
_output_shapes
:?
{rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??k??
yrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/LogLog}rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add:z:0*
T0*
_output_shapes

:?
{rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add_1AddV2?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Const:output:0}rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Log:y:0*
T0*
_output_shapes

:?
yrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/subSub}rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/mul:z:0rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add_1:z:0*
T0*"
_output_shapes
:?
{rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"?????????
irank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/SumSum}rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/sub:z:0?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Sum/reduction_indices:output:0*
T0*
_output_shapes
:?
jrank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/inverse/output_shapeConst*
_output_shapes
:*
dtype0*)
value B"               ?
erank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/inverse/ReshapeReshapetrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Reshape:output:0srank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/inverse/output_shape:output:0*
T0**
_output_shapes
:?
~rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/event_shape_tensor/ConstConst*
_output_shapes
:*
dtype0*%
valueB"            ?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/event_shape_tensor/event_shapeIdentity?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/event_shape_tensor/Const:output:0*
T0*
_output_shapes
:?
yrank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_event_shape_tensor/IdentityIdentity?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/event_shape_tensor/event_shape:output:0*
T0*
_output_shapes
:?
}rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_event_shape_tensor/output_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
|rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"               ?
vrank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/ReshapeReshapenrank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/inverse/Reshape:output:0?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Reshape/shape:output:0*
T0**
_output_shapes
:?
}rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                ?
xrank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/transpose	Transposerank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Reshape:output:0?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/transpose/perm:output:0*
T0**
_output_shapes
:?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOpReadVariableOp?rank_similarity_embedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/SoftplusSoftplus?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/addAddV2?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add/x:output:0?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus:activations:0*
T0*
_output_shapes

:?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/truedivRealDiv|rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/transpose:y:0?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add:z:0*
T0**
_output_shapes
:?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOpReadVariableOpfrank_similarity_embedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_118612*
_output_shapes

:*
dtype0?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/truediv_1RealDiv?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp:value:0?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add:z:0*
T0*
_output_shapes

:?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/SquaredDifferenceSquaredDifference?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/truediv:z:0?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/truediv_1:z:0*
T0**
_output_shapes
:?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/mulMul?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/mul/x:output:0?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/SquaredDifference:z:0*
T0**
_output_shapes
:?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??k??
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/LogLog?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add:z:0*
T0*
_output_shapes

:?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add_1AddV2?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Const:output:0?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Log:y:0*
T0*
_output_shapes

:?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/subSub?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/mul:z:0?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add_1:z:0*
T0**
_output_shapes
:?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"?????????
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/SumSum?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/sub:z:0?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Sum/reduction_indices:output:0*
T0*"
_output_shapes
:?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/BroadcastTo/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
zrank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/BroadcastToBroadcastTo?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Sum:output:0?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/BroadcastTo/shape:output:0*
T0*"
_output_shapes
:?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
rrank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/SumSum?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/BroadcastTo:output:0?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Sum/reduction_indices:output:0*
T0*
_output_shapes
:?
trank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
rrank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/NegNeg}rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/zeros:output:0*
T0*
_output_shapes
: ?
trank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Neg_1Negvrank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Neg:y:0*
T0*
_output_shapes
: ?
srank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/onesConst*
_output_shapes

:*
dtype0*
valueB*  ???
rrank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/mulMul|rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/ones:output:0xrank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Neg_1:y:0*
T0*
_output_shapes

:?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"?????????
rrank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/SumSumvrank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/mul:z:0?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Sum/reduction_indices:output:0*
T0*
_output_shapes
: ?
Qrank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/subSub{rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Sum:output:0{rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Sum:output:0*
T0*
_output_shapes
:?
)rank_similarity/embedding_variational/subSubrrank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Sum:output:0Urank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/sub:z:0*
T0*
_output_shapes
:u
+rank_similarity/embedding_variational/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
*rank_similarity/embedding_variational/MeanMean-rank_similarity/embedding_variational/sub:z:04rank_similarity/embedding_variational/Const:output:0*
T0*
_output_shapes
: p
+rank_similarity/embedding_variational/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *x?V9?
)rank_similarity/embedding_variational/mulMul3rank_similarity/embedding_variational/Mean:output:04rank_similarity/embedding_variational/mul/y:output:0*
T0*
_output_shapes
: ?
:rank_similarity/embedding_variational/mul_1/ReadVariableOpReadVariableOpCrank_similarity_embedding_variational_mul_1_readvariableop_resource*
_output_shapes
: *
dtype0?
+rank_similarity/embedding_variational/mul_1Mul-rank_similarity/embedding_variational/mul:z:0Brank_similarity/embedding_variational/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
: Z
rank_similarity/packed/0Const*
_output_shapes
: *
dtype0*
value	B :?
rank_similarity/packedPack!rank_similarity/packed/0:output:0rank_similarity_packed_values_1*
N*
T0*
_output_shapes
:a
rank_similarity/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
rank_similarity/splitSplitVZrank_similarity/embedding_variational/embedding_normal_diag/Normal/sample/Reshape:output:0rank_similarity/packed:output:0(rank_similarity/split/split_dim:output:0*
T0*

Tlen0*J
_output_shapes8
6:?????????:?????????*
	num_split?
,rank_similarity/distance_based/minkowski/subSubrank_similarity/split:output:0rank_similarity/split:output:1*
T0*/
_output_shapes
:??????????
.rank_similarity/distance_based/minkowski/ShapeShape0rank_similarity/distance_based/minkowski/sub:z:0*
T0*
_output_shapes
:?
<rank_similarity/distance_based/minkowski/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
>rank_similarity/distance_based/minkowski/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
>rank_similarity/distance_based/minkowski/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
6rank_similarity/distance_based/minkowski/strided_sliceStridedSlice7rank_similarity/distance_based/minkowski/Shape:output:0Erank_similarity/distance_based/minkowski/strided_slice/stack:output:0Grank_similarity/distance_based/minkowski/strided_slice/stack_1:output:0Grank_similarity/distance_based/minkowski/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:x
3rank_similarity/distance_based/minkowski/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
-rank_similarity/distance_based/minkowski/onesFill?rank_similarity/distance_based/minkowski/strided_slice:output:0<rank_similarity/distance_based/minkowski/ones/Const:output:0*
T0*+
_output_shapes
:??????????
7rank_similarity/distance_based/minkowski/ReadVariableOpReadVariableOp@rank_similarity_distance_based_minkowski_readvariableop_resource*
_output_shapes
: *
dtype0?
,rank_similarity/distance_based/minkowski/mulMul?rank_similarity/distance_based/minkowski/ReadVariableOp:value:06rank_similarity/distance_based/minkowski/ones:output:0*
T0*+
_output_shapes
:??????????
Crank_similarity/distance_based/minkowski/BroadcastTo/ReadVariableOpReadVariableOpLrank_similarity_distance_based_minkowski_broadcastto_readvariableop_resource*
_output_shapes
:*
dtype0?
4rank_similarity/distance_based/minkowski/BroadcastToBroadcastToKrank_similarity/distance_based/minkowski/BroadcastTo/ReadVariableOp:value:07rank_similarity/distance_based/minkowski/Shape:output:0*
T0*/
_output_shapes
:??????????
,rank_similarity/distance_based/minkowski/AbsAbs0rank_similarity/distance_based/minkowski/sub:z:0*
T0*/
_output_shapes
:??????????
7rank_similarity/distance_based/minkowski/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
3rank_similarity/distance_based/minkowski/ExpandDims
ExpandDims0rank_similarity/distance_based/minkowski/mul:z:0@rank_similarity/distance_based/minkowski/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
,rank_similarity/distance_based/minkowski/PowPow0rank_similarity/distance_based/minkowski/Abs:y:0<rank_similarity/distance_based/minkowski/ExpandDims:output:0*
T0*/
_output_shapes
:??????????
.rank_similarity/distance_based/minkowski/Mul_1Mul0rank_similarity/distance_based/minkowski/Pow:z:0=rank_similarity/distance_based/minkowski/BroadcastTo:output:0*
T0*/
_output_shapes
:??????????
>rank_similarity/distance_based/minkowski/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
,rank_similarity/distance_based/minkowski/SumSum2rank_similarity/distance_based/minkowski/Mul_1:z:0Grank_similarity/distance_based/minkowski/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(w
2rank_similarity/distance_based/minkowski/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
0rank_similarity/distance_based/minkowski/truedivRealDiv;rank_similarity/distance_based/minkowski/truediv/x:output:0<rank_similarity/distance_based/minkowski/ExpandDims:output:0*
T0*/
_output_shapes
:??????????
.rank_similarity/distance_based/minkowski/Pow_1Pow5rank_similarity/distance_based/minkowski/Sum:output:04rank_similarity/distance_based/minkowski/truediv:z:0*
T0*/
_output_shapes
:??????????
1rank_similarity/distance_based/minkowski/IdentityIdentity2rank_similarity/distance_based/minkowski/Pow_1:z:0*
T0*/
_output_shapes
:??????????
2rank_similarity/distance_based/minkowski/IdentityN	IdentityN2rank_similarity/distance_based/minkowski/Pow_1:z:00rank_similarity/distance_based/minkowski/sub:z:0=rank_similarity/distance_based/minkowski/BroadcastTo:output:00rank_similarity/distance_based/minkowski/mul:z:0*
T
2*,
_gradient_op_typeCustomGradient-118785*|
_output_shapesj
h:?????????:?????????:?????????:??????????
0rank_similarity/distance_based/minkowski/SqueezeSqueeze;rank_similarity/distance_based/minkowski/IdentityN:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

??????????
Hrank_similarity/distance_based/exponential_similarity/Neg/ReadVariableOpReadVariableOpQrank_similarity_distance_based_exponential_similarity_neg_readvariableop_resource*
_output_shapes
: *
dtype0?
9rank_similarity/distance_based/exponential_similarity/NegNegPrank_similarity/distance_based/exponential_similarity/Neg/ReadVariableOp:value:0*
T0*
_output_shapes
: ?
Hrank_similarity/distance_based/exponential_similarity/Pow/ReadVariableOpReadVariableOpQrank_similarity_distance_based_exponential_similarity_pow_readvariableop_resource*
_output_shapes
: *
dtype0?
9rank_similarity/distance_based/exponential_similarity/PowPow9rank_similarity/distance_based/minkowski/Squeeze:output:0Prank_similarity/distance_based/exponential_similarity/Pow/ReadVariableOp:value:0*
T0*+
_output_shapes
:??????????
9rank_similarity/distance_based/exponential_similarity/mulMul=rank_similarity/distance_based/exponential_similarity/Neg:y:0=rank_similarity/distance_based/exponential_similarity/Pow:z:0*
T0*+
_output_shapes
:??????????
9rank_similarity/distance_based/exponential_similarity/ExpExp=rank_similarity/distance_based/exponential_similarity/mul:z:0*
T0*+
_output_shapes
:??????????
Hrank_similarity/distance_based/exponential_similarity/add/ReadVariableOpReadVariableOpQrank_similarity_distance_based_exponential_similarity_add_readvariableop_resource*
_output_shapes
: *
dtype0?
9rank_similarity/distance_based/exponential_similarity/addAddV2=rank_similarity/distance_based/exponential_similarity/Exp:y:0Prank_similarity/distance_based/exponential_similarity/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????
rank_similarity/CastCastrank_similarity/NotEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:??????????
,rank_similarity/rank_sim_zero_out_nonpresentMul=rank_similarity/distance_based/exponential_similarity/add:z:0rank_similarity/Cast:y:0*
T0*+
_output_shapes
:??????????
rank_similarity/GatherV2_1GatherV20rank_similarity/rank_sim_zero_out_nonpresent:z:0"rank_similarity_gatherv2_1_indicesrank_similarity_gatherv2_axis*
Taxis0*
Tindices0*
Tparams0*/
_output_shapes
:?????????`
rank_similarity/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
rank_similarity/ExpandDims
ExpandDimsrank_similarity/Cast:y:0'rank_similarity/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????W
rank_similarity/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
rank_similarity/GatherV2_2GatherV2#rank_similarity/ExpandDims:output:0rank_similarity/Const:output:0rank_similarity_gatherv2_axis*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:?????????]
rank_similarity/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B :?
rank_similarity/CumsumCumsum#rank_similarity/GatherV2_1:output:0$rank_similarity/Cumsum/axis:output:0*
T0*/
_output_shapes
:?????????*
reverse(^
rank_similarity/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
rank_similarity/MaximumMaximum#rank_similarity/GatherV2_1:output:0"rank_similarity/Maximum/y:output:0*
T0*/
_output_shapes
:?????????`
rank_similarity/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
rank_similarity/Maximum_1Maximumrank_similarity/Cumsum:out:0$rank_similarity/Maximum_1/y:output:0*
T0*/
_output_shapes
:?????????q
rank_similarity/LogLogrank_similarity/Maximum:z:0*
T0*/
_output_shapes
:?????????u
rank_similarity/Log_1Logrank_similarity/Maximum_1:z:0*
T0*/
_output_shapes
:??????????
rank_similarity/subSubrank_similarity/Log:y:0rank_similarity/Log_1:y:0*
T0*/
_output_shapes
:??????????
rank_similarity/mulMulrank_similarity_mul_xrank_similarity/sub:z:0*
T0*/
_output_shapes
:?????????g
%rank_similarity/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
rank_similarity/SumSumrank_similarity/mul:z:0.rank_similarity/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????n
rank_similarity/ExpExprank_similarity/Sum:output:0*
T0*+
_output_shapes
:??????????
rank_similarity/mul_1Mul#rank_similarity/GatherV2_2:output:0rank_similarity/Exp:y:0*
T0*+
_output_shapes
:?????????i
'rank_similarity/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
rank_similarity/Sum_1Sumrank_similarity/mul_1:z:00rank_similarity/Sum_1/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(\
rank_similarity/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
rank_similarity/EqualEqualrank_similarity/Sum_1:output:0 rank_similarity/Equal/y:output:0*
T0*+
_output_shapes
:?????????~
rank_similarity/Cast_1Castrank_similarity/Equal:z:0*

DstT0*

SrcT0
*+
_output_shapes
:??????????
rank_similarity/truedivRealDivrank_similarity/Cast_1:y:0rank_similarity_truediv_y*
T0*+
_output_shapes
:??????????
rank_similarity/addAddV2rank_similarity/mul_1:z:0rank_similarity/truediv:z:0*
T0*+
_output_shapes
:??????????
rank_similarity/add_1AddV2rank_similarity/Sum_1:output:0rank_similarity/Cast_1:y:0*
T0*+
_output_shapes
:??????????
rank_similarity/truediv_1RealDivrank_similarity/add:z:0rank_similarity/add_1:z:0*
T0*+
_output_shapes
:?????????p
IdentityIdentityrank_similarity/truediv_1:z:0^NoOp*
T0*+
_output_shapes
:?????????o

Identity_1Identity/rank_similarity/embedding_variational/mul_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOpI^rank_similarity/distance_based/exponential_similarity/Neg/ReadVariableOpI^rank_similarity/distance_based/exponential_similarity/Pow/ReadVariableOpI^rank_similarity/distance_based/exponential_similarity/add/ReadVariableOpD^rank_similarity/distance_based/minkowski/BroadcastTo/ReadVariableOp8^rank_similarity/distance_based/minkowski/ReadVariableOp?^rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOp?^rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOp?^rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOp?^rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOpM^rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookupg^rank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOp`^rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookupz^rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOp;^rank_similarity/embedding_variational/mul_1/ReadVariableOp?^rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp?^rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:?????????:?????????:: : : : : : : : : : : : ::: 2?
Hrank_similarity/distance_based/exponential_similarity/Neg/ReadVariableOpHrank_similarity/distance_based/exponential_similarity/Neg/ReadVariableOp2?
Hrank_similarity/distance_based/exponential_similarity/Pow/ReadVariableOpHrank_similarity/distance_based/exponential_similarity/Pow/ReadVariableOp2?
Hrank_similarity/distance_based/exponential_similarity/add/ReadVariableOpHrank_similarity/distance_based/exponential_similarity/add/ReadVariableOp2?
Crank_similarity/distance_based/minkowski/BroadcastTo/ReadVariableOpCrank_similarity/distance_based/minkowski/BroadcastTo/ReadVariableOp2r
7rank_similarity/distance_based/minkowski/ReadVariableOp7rank_similarity/distance_based/minkowski/ReadVariableOp2?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOp?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOp2?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOp?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOp2?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOp?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOp2?
?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOp?rank_similarity/embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOp2?
Lrank_similarity/embedding_variational/embedding_normal_diag/embedding_lookupLrank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup2?
frank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOpfrank_similarity/embedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOp2?
_rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_rank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup2?
yrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOpyrank_similarity/embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOp2x
:rank_similarity/embedding_variational/mul_1/ReadVariableOp:rank_similarity/embedding_variational/mul_1/ReadVariableOp2?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp2?
?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp?rank_similarity/embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp:g c
+
_output_shapes
:?????????
4
_user_specified_nameinputs/2rank1/stimulus_set:\X
+
_output_shapes
:?????????
)
_user_specified_nameinputs/agent_id: 

_output_shapes
::

_output_shapes
: :	

_output_shapes
: :$ 

_output_shapes

::,(
&
_output_shapes
::

_output_shapes
: 
??
?
K__inference_rank_similarity_layer_call_and_return_conditional_losses_118157

inputs
inputs_1
gatherv2_indices
gatherv2_axisU
Cembedding_variational_embedding_normal_diag_embedding_lookup_117872:q
_embedding_variational_embedding_normal_diag_embedding_lookup_1_softplus_readvariableop_resource:h
Vembedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_117921:?
rembedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_1_softplus_readvariableop_resource:=
3embedding_variational_mul_1_readvariableop_resource: 
packed_values_1:
0distance_based_minkowski_readvariableop_resource: J
<distance_based_minkowski_broadcastto_readvariableop_resource:K
Adistance_based_exponential_similarity_neg_readvariableop_resource: K
Adistance_based_exponential_similarity_pow_readvariableop_resource: K
Adistance_based_exponential_similarity_add_readvariableop_resource: 
gatherv2_1_indices	
mul_x
	truediv_y
identity

identity_1??8distance_based/exponential_similarity/Neg/ReadVariableOp?8distance_based/exponential_similarity/Pow/ReadVariableOp?8distance_based/exponential_similarity/add/ReadVariableOp?3distance_based/minkowski/BroadcastTo/ReadVariableOp?'distance_based/minkowski/ReadVariableOp?tembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOp?}embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOp?pembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOp?yembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOp?<embedding_variational/embedding_normal_diag/embedding_lookup?Vembedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOp?Oembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup?iembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOp?*embedding_variational/mul_1/ReadVariableOp??embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp??embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp?
GatherV2GatherV2inputsgatherv2_indicesgatherv2_axis*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:?????????L

NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : r
NotEqualNotEqualGatherV2:output:0NotEqual/y:output:0*
T0*+
_output_shapes
:??????????
<embedding_variational/embedding_normal_diag/embedding_lookupResourceGatherCembedding_variational_embedding_normal_diag_embedding_lookup_117872inputs*
Tindices0*V
_classL
JHloc:@embedding_variational/embedding_normal_diag/embedding_lookup/117872*/
_output_shapes
:?????????*
dtype0?
Eembedding_variational/embedding_normal_diag/embedding_lookup/IdentityIdentityEembedding_variational/embedding_normal_diag/embedding_lookup:output:0*
T0*V
_classL
JHloc:@embedding_variational/embedding_normal_diag/embedding_lookup/117872*/
_output_shapes
:??????????
Gembedding_variational/embedding_normal_diag/embedding_lookup/Identity_1IdentityNembedding_variational/embedding_normal_diag/embedding_lookup/Identity:output:0*
T0*/
_output_shapes
:??????????
Vembedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOpReadVariableOp_embedding_variational_embedding_normal_diag_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
Gembedding_variational/embedding_normal_diag/embedding_lookup_1/SoftplusSoftplus^embedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
Dembedding_variational/embedding_normal_diag/embedding_lookup_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
Bembedding_variational/embedding_normal_diag/embedding_lookup_1/addAddV2Membedding_variational/embedding_normal_diag/embedding_lookup_1/add/x:output:0Uembedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus:activations:0*
T0*
_output_shapes

:?
Cembedding_variational/embedding_normal_diag/embedding_lookup_1/axisConst*U
_classK
IGloc:@embedding_variational/embedding_normal_diag/embedding_lookup_1/add*
_output_shapes
: *
dtype0*
value	B : ?
>embedding_variational/embedding_normal_diag/embedding_lookup_1GatherV2Fembedding_variational/embedding_normal_diag/embedding_lookup_1/add:z:0inputsLembedding_variational/embedding_normal_diag/embedding_lookup_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*U
_classK
IGloc:@embedding_variational/embedding_normal_diag/embedding_lookup_1/add*/
_output_shapes
:??????????
Gembedding_variational/embedding_normal_diag/embedding_lookup_1/IdentityIdentityGembedding_variational/embedding_normal_diag/embedding_lookup_1:output:0*
T0*/
_output_shapes
:??????????
Fembedding_variational/embedding_normal_diag/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
?embedding_variational/embedding_normal_diag/Normal/sample/ShapeShapePembedding_variational/embedding_normal_diag/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
?embedding_variational/embedding_normal_diag/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
Membedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Oembedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Oembedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Gembedding_variational/embedding_normal_diag/Normal/sample/strided_sliceStridedSliceHembedding_variational/embedding_normal_diag/Normal/sample/Shape:output:0Vembedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack:output:0Xembedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack_1:output:0Xembedding_variational/embedding_normal_diag/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Aembedding_variational/embedding_normal_diag/Normal/sample/Shape_1ShapePembedding_variational/embedding_normal_diag/embedding_lookup_1/Identity:output:0*
T0*
_output_shapes
:?
Aembedding_variational/embedding_normal_diag/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : ?
Oembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Qembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Qembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Iembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1StridedSliceJembedding_variational/embedding_normal_diag/Normal/sample/Shape_1:output:0Xembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack:output:0Zembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack_1:output:0Zembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Jembedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
Lembedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
Gembedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgsBroadcastArgsUembedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs/s0_1:output:0Pembedding_variational/embedding_normal_diag/Normal/sample/strided_slice:output:0*
_output_shapes
:?
Iembedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs_1BroadcastArgsLembedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs:r0:0Rembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_1:output:0*
_output_shapes
:?
Iembedding_variational/embedding_normal_diag/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
Eembedding_variational/embedding_normal_diag/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
@embedding_variational/embedding_normal_diag/Normal/sample/concatConcatV2Rembedding_variational/embedding_normal_diag/Normal/sample/concat/values_0:output:0Nembedding_variational/embedding_normal_diag/Normal/sample/BroadcastArgs_1:r0:0Nembedding_variational/embedding_normal_diag/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Sembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
Uembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
cembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalIembedding_variational/embedding_normal_diag/Normal/sample/concat:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????*
dtype0?
Rembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/mulMullembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/RandomStandardNormal:output:0^embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/stddev:output:0*
T0*N
_output_shapes<
::8?????????????????????????????????????
Nembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normalAddV2Vembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/mul:z:0\embedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal/mean:output:0*
T0*N
_output_shapes<
::8?????????????????????????????????????
=embedding_variational/embedding_normal_diag/Normal/sample/mulMulRembedding_variational/embedding_normal_diag/Normal/sample/normal/random_normal:z:0Pembedding_variational/embedding_normal_diag/embedding_lookup_1/Identity:output:0*
T0*E
_output_shapes3
1:/????????????????????????????
=embedding_variational/embedding_normal_diag/Normal/sample/addAddV2Aembedding_variational/embedding_normal_diag/Normal/sample/mul:z:0Pembedding_variational/embedding_normal_diag/embedding_lookup/Identity_1:output:0*
T0*E
_output_shapes3
1:/????????????????????????????
Aembedding_variational/embedding_normal_diag/Normal/sample/Shape_2ShapeAembedding_variational/embedding_normal_diag/Normal/sample/add:z:0*
T0*
_output_shapes
:?
Oembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Qembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Qembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Iembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2StridedSliceJembedding_variational/embedding_normal_diag/Normal/sample/Shape_2:output:0Xembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack:output:0Zembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack_1:output:0Zembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Gembedding_variational/embedding_normal_diag/Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Bembedding_variational/embedding_normal_diag/Normal/sample/concat_1ConcatV2Oembedding_variational/embedding_normal_diag/Normal/sample/sample_shape:output:0Rembedding_variational/embedding_normal_diag/Normal/sample/strided_slice_2:output:0Pembedding_variational/embedding_normal_diag/Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Aembedding_variational/embedding_normal_diag/Normal/sample/ReshapeReshapeAembedding_variational/embedding_normal_diag/Normal/sample/add:z:0Kembedding_variational/embedding_normal_diag/Normal/sample/concat_1:output:0*
T0*/
_output_shapes
:?????????|
1embedding_variational/embedding_shared/zeros_like	ZerosLikeinputs*
T0*+
_output_shapes
:??????????
Oembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookupResourceGatherVembedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_1179215embedding_variational/embedding_shared/zeros_like:y:0*
Tindices0*i
_class_
][loc:@embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/117921*/
_output_shapes
:?????????*
dtype0?
Xembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/IdentityIdentityXembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup:output:0*
T0*i
_class_
][loc:@embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/117921*/
_output_shapes
:??????????
Zembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/Identity_1Identityaembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/Identity:output:0*
T0*/
_output_shapes
:??????????
iembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOpReadVariableOprembedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
Zembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/SoftplusSoftplusqembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
Wembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
Uembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/addAddV2`embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add/x:output:0hembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus:activations:0*
T0*
_output_shapes

:?
Vembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/axisConst*h
_class^
\Zloc:@embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add*
_output_shapes
: *
dtype0*
value	B : ?
Qembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1GatherV2Yembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add:z:05embedding_variational/embedding_shared/zeros_like:y:0_embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*h
_class^
\Zloc:@embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/add*/
_output_shapes
:??????????
Zembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/IdentityIdentityZembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1:output:0*
T0*/
_output_shapes
:??????????
Yembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Rembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/ShapeShapecembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
Rembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
`embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
bembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
bembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Zembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_sliceStridedSlice[embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape:output:0iembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack:output:0kembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack_1:output:0kembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Tembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape_1Shapecembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Identity:output:0*
T0*
_output_shapes
:?
Tembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : ?
bembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
dembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
dembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1StridedSlice]embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape_1:output:0kembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack:output:0membedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack_1:output:0membedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
]embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
_embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
Zembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgsBroadcastArgshembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs/s0_1:output:0cembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice:output:0*
_output_shapes
:?
\embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs_1BroadcastArgs_embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs:r0:0eembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_1:output:0*
_output_shapes
:?
\embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
Xembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Sembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concatConcatV2eembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat/values_0:output:0aembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/BroadcastArgs_1:r0:0aembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
fembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
hembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
vembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal\embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat:output:0*
T0*N
_output_shapes<
::8????????????????????????????????????*
dtype0?
eembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/mulMulembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/RandomStandardNormal:output:0qembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/stddev:output:0*
T0*N
_output_shapes<
::8?????????????????????????????????????
aembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normalAddV2iembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/mul:z:0oembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal/mean:output:0*
T0*N
_output_shapes<
::8?????????????????????????????????????
Pembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/mulMuleembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/normal/random_normal:z:0cembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Identity:output:0*
T0*E
_output_shapes3
1:/????????????????????????????
Pembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/addAddV2Tembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/mul:z:0cembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup/Identity_1:output:0*
T0*E
_output_shapes3
1:/????????????????????????????
Tembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape_2ShapeTembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/add:z:0*
T0*
_output_shapes
:?
bembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
dembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
\embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2StridedSlice]embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/Shape_2:output:0kembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack:output:0membedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack_1:output:0membedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Zembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Uembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat_1ConcatV2bembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/sample_shape:output:0eembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/strided_slice_2:output:0cembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Tembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/ReshapeReshapeTembedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/add:z:0^embedding_variational/embedding_shared/embedding_normal_diag_1/Normal/sample/concat_1:output:0*
T0*/
_output_shapes
:?????????p
-embedding_variational/reshape/event_shape_outConst*
_output_shapes
: *
dtype0*
valueB }
,embedding_variational/reshape/event_shape_inConst*
_output_shapes
:*
dtype0*
valueB"      ?
:embedding_variational/SampleIndependentNormal/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB"      {
9embedding_variational/reshapeSampleIndependentNormal/zeroConst*
_output_shapes
: *
dtype0*
value	B : ?
`embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
qembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
nembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
pembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOpReadVariableOpCembedding_variational_embedding_normal_diag_embedding_lookup_117872*
_output_shapes

:*
dtype0?
yembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOpReadVariableOp_embedding_variational_embedding_normal_diag_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
jembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/SoftplusSoftplus?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
gembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
eembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/addAddV2pembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add/x:output:0xembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus:activations:0*
T0*
_output_shapes

:?
qembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"      ?
gembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
uembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
wembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
wembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
oembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_sliceStridedSlicezembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/shape_as_tensor:output:0~embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack:output:0?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack_1:output:0?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
sembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"      ?
iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : ?
wembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
yembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
yembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
qembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1StridedSlice|embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/shape_as_tensor_1:output:0?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack:output:0?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack_1:output:0?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
rembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
tembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
oembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgsBroadcastArgs}embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs/s0_1:output:0xembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice:output:0*
_output_shapes
:?
qembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs_1BroadcastArgstembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs:r0:0zembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/strided_slice_1:output:0*
_output_shapes
:?
qembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
membedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
hembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concatConcatV2zembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat/values_0:output:0vembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/BroadcastArgs_1:r0:0vembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
{embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
}embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalqembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/concat:output:0*
T0*"
_output_shapes
:*
dtype0?
zembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/mulMul?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/RandomStandardNormal:output:0?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/stddev:output:0*
T0*"
_output_shapes
:?
vembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normalAddV2~embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/mul:z:0?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal/mean:output:0*
T0*"
_output_shapes
:?
eembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/mulMulzembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/normal/random_normal:z:0iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add:z:0*
T0*"
_output_shapes
:?
gembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add_1AddV2iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/mul:z:0xembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOp:value:0*
T0*"
_output_shapes
:?
oembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReshapeReshapekembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/add_1:z:0xembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Reshape/shape:output:0*
T0*"
_output_shapes
:?
aembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
[embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/ReshapeReshaperembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Reshape:output:0jembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Reshape/shape:output:0*
T0*"
_output_shapes
:?
}embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOpReadVariableOp_embedding_variational_embedding_normal_diag_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
nembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/SoftplusSoftplus?embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
kembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/addAddV2tembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add/x:output:0|embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus:activations:0*
T0*
_output_shapes

:?
membedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/truedivRealDivdembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Reshape:output:0membedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add:z:0*
T0*"
_output_shapes
:?
tembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOpReadVariableOpCembedding_variational_embedding_normal_diag_embedding_lookup_117872*
_output_shapes

:*
dtype0?
oembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/truediv_1RealDiv|embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOp:value:0membedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add:z:0*
T0*
_output_shapes

:?
wembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/SquaredDifferenceSquaredDifferenceqembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/truediv:z:0sembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/truediv_1:z:0*
T0*"
_output_shapes
:?
kembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/mulMultembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/mul/x:output:0{embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/SquaredDifference:z:0*
T0*"
_output_shapes
:?
kembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??k??
iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/LogLogmembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add:z:0*
T0*
_output_shapes

:?
kembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add_1AddV2tembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Const:output:0membedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Log:y:0*
T0*
_output_shapes

:?
iembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/subSubmembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/mul:z:0oembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/add_1:z:0*
T0*"
_output_shapes
:?
kembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"?????????
Yembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/SumSummembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/sub:z:0tembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Sum/reduction_indices:output:0*
T0*
_output_shapes
:?
Zembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/inverse/output_shapeConst*
_output_shapes
:*
dtype0*)
value B"               ?
Uembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/inverse/ReshapeReshapedembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Reshape:output:0cembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/inverse/output_shape:output:0*
T0**
_output_shapes
:?
nembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/event_shape_tensor/ConstConst*
_output_shapes
:*
dtype0*%
valueB"            ?
tembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/event_shape_tensor/event_shapeIdentitywembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/event_shape_tensor/Const:output:0*
T0*
_output_shapes
:?
iembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_event_shape_tensor/IdentityIdentity}embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/event_shape_tensor/event_shape:output:0*
T0*
_output_shapes
:?
membedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_event_shape_tensor/output_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
lembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Reshape/shapeConst*
_output_shapes
:*
dtype0*)
value B"               ?
fembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/ReshapeReshape^embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/inverse/Reshape:output:0uembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Reshape/shape:output:0*
T0**
_output_shapes
:?
membedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                ?
hembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/transpose	Transposeoembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Reshape:output:0vembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/transpose/perm:output:0*
T0**
_output_shapes
:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOpReadVariableOprembedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_1_softplus_readvariableop_resource*
_output_shapes

:*
dtype0?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/SoftplusSoftplus?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/addAddV2?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add/x:output:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus:activations:0*
T0*
_output_shapes

:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/truedivRealDivlembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/transpose:y:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add:z:0*
T0**
_output_shapes
:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOpReadVariableOpVembedding_variational_embedding_shared_embedding_normal_diag_1_embedding_lookup_117921*
_output_shapes

:*
dtype0?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/truediv_1RealDiv?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp:value:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add:z:0*
T0*
_output_shapes

:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/SquaredDifferenceSquaredDifference?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/truediv:z:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/truediv_1:z:0*
T0**
_output_shapes
:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/mulMul?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/mul/x:output:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/SquaredDifference:z:0*
T0**
_output_shapes
:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??k??
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/LogLog?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add:z:0*
T0*
_output_shapes

:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add_1AddV2?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Const:output:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Log:y:0*
T0*
_output_shapes

:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/subSub?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/mul:z:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/add_1:z:0*
T0**
_output_shapes
:?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"?????????
}embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/SumSum?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/sub:z:0?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Sum/reduction_indices:output:0*
T0*"
_output_shapes
:?
pembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/BroadcastTo/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ?
jembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/BroadcastToBroadcastTo?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Sum:output:0yembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/BroadcastTo/shape:output:0*
T0*"
_output_shapes
:?
tembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
bembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/SumSumsembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/BroadcastTo:output:0}embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Sum/reduction_indices:output:0*
T0*
_output_shapes
:?
dembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
bembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/NegNegmembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/zeros:output:0*
T0*
_output_shapes
: ?
dembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Neg_1Negfembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Neg:y:0*
T0*
_output_shapes
: ?
cembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/onesConst*
_output_shapes

:*
dtype0*
valueB*  ???
bembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/mulMullembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/ones:output:0hembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Neg_1:y:0*
T0*
_output_shapes

:?
tembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"?????????
bembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/SumSumfembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/mul:z:0}embedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Sum/reduction_indices:output:0*
T0*
_output_shapes
: ?
Aembedding_variational/reshapeSampleIndependentNormal/log_prob/subSubkembedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/Sum:output:0kembedding_variational/reshapeSampleIndependentNormal/log_prob/reshape/forward_log_det_jacobian/Sum:output:0*
T0*
_output_shapes
:?
embedding_variational/subSubbembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Sum:output:0Eembedding_variational/reshapeSampleIndependentNormal/log_prob/sub:z:0*
T0*
_output_shapes
:e
embedding_variational/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
embedding_variational/MeanMeanembedding_variational/sub:z:0$embedding_variational/Const:output:0*
T0*
_output_shapes
: `
embedding_variational/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *x?V9?
embedding_variational/mulMul#embedding_variational/Mean:output:0$embedding_variational/mul/y:output:0*
T0*
_output_shapes
: ?
*embedding_variational/mul_1/ReadVariableOpReadVariableOp3embedding_variational_mul_1_readvariableop_resource*
_output_shapes
: *
dtype0?
embedding_variational/mul_1Mulembedding_variational/mul:z:02embedding_variational/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
: J
packed/0Const*
_output_shapes
: *
dtype0*
value	B :`
packedPackpacked/0:output:0packed_values_1*
N*
T0*
_output_shapes
:Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
splitSplitVJembedding_variational/embedding_normal_diag/Normal/sample/Reshape:output:0packed:output:0split/split_dim:output:0*
T0*

Tlen0*J
_output_shapes8
6:?????????:?????????*
	num_split}
distance_based/minkowski/subSubsplit:output:0split:output:1*
T0*/
_output_shapes
:?????????n
distance_based/minkowski/ShapeShape distance_based/minkowski/sub:z:0*
T0*
_output_shapes
:v
,distance_based/minkowski/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
.distance_based/minkowski/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????x
.distance_based/minkowski/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&distance_based/minkowski/strided_sliceStridedSlice'distance_based/minkowski/Shape:output:05distance_based/minkowski/strided_slice/stack:output:07distance_based/minkowski/strided_slice/stack_1:output:07distance_based/minkowski/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:h
#distance_based/minkowski/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
distance_based/minkowski/onesFill/distance_based/minkowski/strided_slice:output:0,distance_based/minkowski/ones/Const:output:0*
T0*+
_output_shapes
:??????????
'distance_based/minkowski/ReadVariableOpReadVariableOp0distance_based_minkowski_readvariableop_resource*
_output_shapes
: *
dtype0?
distance_based/minkowski/mulMul/distance_based/minkowski/ReadVariableOp:value:0&distance_based/minkowski/ones:output:0*
T0*+
_output_shapes
:??????????
3distance_based/minkowski/BroadcastTo/ReadVariableOpReadVariableOp<distance_based_minkowski_broadcastto_readvariableop_resource*
_output_shapes
:*
dtype0?
$distance_based/minkowski/BroadcastToBroadcastTo;distance_based/minkowski/BroadcastTo/ReadVariableOp:value:0'distance_based/minkowski/Shape:output:0*
T0*/
_output_shapes
:?????????
distance_based/minkowski/AbsAbs distance_based/minkowski/sub:z:0*
T0*/
_output_shapes
:?????????r
'distance_based/minkowski/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
#distance_based/minkowski/ExpandDims
ExpandDims distance_based/minkowski/mul:z:00distance_based/minkowski/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
distance_based/minkowski/PowPow distance_based/minkowski/Abs:y:0,distance_based/minkowski/ExpandDims:output:0*
T0*/
_output_shapes
:??????????
distance_based/minkowski/Mul_1Mul distance_based/minkowski/Pow:z:0-distance_based/minkowski/BroadcastTo:output:0*
T0*/
_output_shapes
:?????????y
.distance_based/minkowski/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
distance_based/minkowski/SumSum"distance_based/minkowski/Mul_1:z:07distance_based/minkowski/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(g
"distance_based/minkowski/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
 distance_based/minkowski/truedivRealDiv+distance_based/minkowski/truediv/x:output:0,distance_based/minkowski/ExpandDims:output:0*
T0*/
_output_shapes
:??????????
distance_based/minkowski/Pow_1Pow%distance_based/minkowski/Sum:output:0$distance_based/minkowski/truediv:z:0*
T0*/
_output_shapes
:??????????
!distance_based/minkowski/IdentityIdentity"distance_based/minkowski/Pow_1:z:0*
T0*/
_output_shapes
:??????????
"distance_based/minkowski/IdentityN	IdentityN"distance_based/minkowski/Pow_1:z:0 distance_based/minkowski/sub:z:0-distance_based/minkowski/BroadcastTo:output:0 distance_based/minkowski/mul:z:0*
T
2*,
_gradient_op_typeCustomGradient-118094*|
_output_shapesj
h:?????????:?????????:?????????:??????????
 distance_based/minkowski/SqueezeSqueeze+distance_based/minkowski/IdentityN:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

??????????
8distance_based/exponential_similarity/Neg/ReadVariableOpReadVariableOpAdistance_based_exponential_similarity_neg_readvariableop_resource*
_output_shapes
: *
dtype0?
)distance_based/exponential_similarity/NegNeg@distance_based/exponential_similarity/Neg/ReadVariableOp:value:0*
T0*
_output_shapes
: ?
8distance_based/exponential_similarity/Pow/ReadVariableOpReadVariableOpAdistance_based_exponential_similarity_pow_readvariableop_resource*
_output_shapes
: *
dtype0?
)distance_based/exponential_similarity/PowPow)distance_based/minkowski/Squeeze:output:0@distance_based/exponential_similarity/Pow/ReadVariableOp:value:0*
T0*+
_output_shapes
:??????????
)distance_based/exponential_similarity/mulMul-distance_based/exponential_similarity/Neg:y:0-distance_based/exponential_similarity/Pow:z:0*
T0*+
_output_shapes
:??????????
)distance_based/exponential_similarity/ExpExp-distance_based/exponential_similarity/mul:z:0*
T0*+
_output_shapes
:??????????
8distance_based/exponential_similarity/add/ReadVariableOpReadVariableOpAdistance_based_exponential_similarity_add_readvariableop_resource*
_output_shapes
: *
dtype0?
)distance_based/exponential_similarity/addAddV2-distance_based/exponential_similarity/Exp:y:0@distance_based/exponential_similarity/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????_
CastCastNotEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:??????????
rank_sim_zero_out_nonpresentMul-distance_based/exponential_similarity/add:z:0Cast:y:0*
T0*+
_output_shapes
:??????????

GatherV2_1GatherV2 rank_sim_zero_out_nonpresent:z:0gatherv2_1_indicesgatherv2_axis*
Taxis0*
Tindices0*
Tparams0*/
_output_shapes
:?????????P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :u

ExpandDims
ExpandDimsCast:y:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????G
ConstConst*
_output_shapes
: *
dtype0*
value	B : ?

GatherV2_2GatherV2ExpandDims:output:0Const:output:0gatherv2_axis*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:?????????M
Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B :?
CumsumCumsumGatherV2_1:output:0Cumsum/axis:output:0*
T0*/
_output_shapes
:?????????*
reverse(N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3u
MaximumMaximumGatherV2_1:output:0Maximum/y:output:0*
T0*/
_output_shapes
:?????????P
Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3r
	Maximum_1MaximumCumsum:out:0Maximum_1/y:output:0*
T0*/
_output_shapes
:?????????Q
LogLogMaximum:z:0*
T0*/
_output_shapes
:?????????U
Log_1LogMaximum_1:z:0*
T0*/
_output_shapes
:?????????X
subSubLog:y:0	Log_1:y:0*
T0*/
_output_shapes
:?????????T
mulMulmul_xsub:z:0*
T0*/
_output_shapes
:?????????W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :i
SumSummul:z:0Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????N
ExpExpSum:output:0*
T0*+
_output_shapes
:?????????`
mul_1MulGatherV2_2:output:0Exp:y:0*
T0*+
_output_shapes
:?????????Y
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
Sum_1Sum	mul_1:z:0 Sum_1/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(L
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *    f
EqualEqualSum_1:output:0Equal/y:output:0*
T0*+
_output_shapes
:?????????^
Cast_1Cast	Equal:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????_
truedivRealDiv
Cast_1:y:0	truediv_y*
T0*+
_output_shapes
:?????????Z
addAddV2	mul_1:z:0truediv:z:0*
T0*+
_output_shapes
:?????????`
add_1AddV2Sum_1:output:0
Cast_1:y:0*
T0*+
_output_shapes
:?????????^
	truediv_1RealDivadd:z:0	add_1:z:0*
T0*+
_output_shapes
:?????????`
IdentityIdentitytruediv_1:z:0^NoOp*
T0*+
_output_shapes
:?????????_

Identity_1Identityembedding_variational/mul_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp9^distance_based/exponential_similarity/Neg/ReadVariableOp9^distance_based/exponential_similarity/Pow/ReadVariableOp9^distance_based/exponential_similarity/add/ReadVariableOp4^distance_based/minkowski/BroadcastTo/ReadVariableOp(^distance_based/minkowski/ReadVariableOpu^embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOp~^embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOpq^embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOpz^embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOp=^embedding_variational/embedding_normal_diag/embedding_lookupW^embedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOpP^embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookupj^embedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOp+^embedding_variational/mul_1/ReadVariableOp?^embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp?^embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:?????????:?????????:: : : : : : : : : : : : ::: 2t
8distance_based/exponential_similarity/Neg/ReadVariableOp8distance_based/exponential_similarity/Neg/ReadVariableOp2t
8distance_based/exponential_similarity/Pow/ReadVariableOp8distance_based/exponential_similarity/Pow/ReadVariableOp2t
8distance_based/exponential_similarity/add/ReadVariableOp8distance_based/exponential_similarity/add/ReadVariableOp2j
3distance_based/minkowski/BroadcastTo/ReadVariableOp3distance_based/minkowski/BroadcastTo/ReadVariableOp2R
'distance_based/minkowski/ReadVariableOp'distance_based/minkowski/ReadVariableOp2?
tembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOptembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/ReadVariableOp2?
}embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOp}embedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/log_prob/Normal/log_prob/Softplus/ReadVariableOp2?
pembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOppembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/ReadVariableOp2?
yembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOpyembedding_variational/IndependentNormal_CONSTRUCTED_AT_embedding_normal_diag/sample/Normal/sample/Softplus/ReadVariableOp2|
<embedding_variational/embedding_normal_diag/embedding_lookup<embedding_variational/embedding_normal_diag/embedding_lookup2?
Vembedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOpVembedding_variational/embedding_normal_diag/embedding_lookup_1/Softplus/ReadVariableOp2?
Oembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookupOembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup2?
iembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOpiembedding_variational/embedding_shared/embedding_normal_diag_1/embedding_lookup_1/Softplus/ReadVariableOp2X
*embedding_variational/mul_1/ReadVariableOp*embedding_variational/mul_1/ReadVariableOp2?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/ReadVariableOp2?
?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp?embedding_variational/reshapeSampleIndependentNormal/log_prob/SampleIndependentNormal/log_prob/IndependentNormal/log_prob/Normal/log_prob/Softplus/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????
 
_user_specified_nameinputs: 

_output_shapes
::

_output_shapes
: :	

_output_shapes
: :$ 

_output_shapes

::,(
&
_output_shapes
::

_output_shapes
: 
?7
?
#__inference_internal_grad_fn_120177
result_grads_0
result_grads_1
result_grads_2
result_grads_3<
8mul_rank_similarity_distance_based_minkowski_broadcastto4
0mul_rank_similarity_distance_based_minkowski_sub4
0pow_rank_similarity_distance_based_minkowski_abs;
7div_no_nan_rank_similarity_distance_based_minkowski_pow;
7sub_rank_similarity_distance_based_minkowski_expanddims8
4pow_1_rank_similarity_distance_based_minkowski_pow_1=
9div_no_nan_1_rank_similarity_distance_based_minkowski_sum8
4mul_6_rank_similarity_distance_based_minkowski_mul_1
identity

identity_1

identity_2?
mulMul8mul_rank_similarity_distance_based_minkowski_broadcastto0mul_rank_similarity_distance_based_minkowski_sub*
T0*/
_output_shapes
:?????????J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
powPow0pow_rank_similarity_distance_based_minkowski_abspow/y:output:0*
T0*/
_output_shapes
:??????????

div_no_nanDivNoNan7div_no_nan_rank_similarity_distance_based_minkowski_powpow:z:0*
T0*/
_output_shapes
:?????????_
mul_1Mulmul:z:0div_no_nan:z:0*
T0*/
_output_shapes
:?????????J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
subSub7sub_rank_similarity_distance_based_minkowski_expanddimssub/y:output:0*
T0*/
_output_shapes
:??????????
pow_1Pow4pow_1_rank_similarity_distance_based_minkowski_pow_1sub:z:0*
T0*/
_output_shapes
:?????????J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3a
addAddV2	pow_1:z:0add/y:output:0*
T0*/
_output_shapes
:?????????`
truedivRealDiv	mul_1:z:0add:z:0*
T0*/
_output_shapes
:?????????c
mul_2Mulresult_grads_0truediv:z:0*
T0*/
_output_shapes
:?????????L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
sub_1Sub7sub_rank_similarity_distance_based_minkowski_expanddimssub_1/y:output:0*
T0*/
_output_shapes
:??????????
pow_2Pow4pow_1_rank_similarity_distance_based_minkowski_pow_1	sub_1:z:0*
T0*/
_output_shapes
:??????????
mul_3Mul7sub_rank_similarity_distance_based_minkowski_expanddims	pow_2:z:0*
T0*/
_output_shapes
:?????????L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3e
add_1AddV2	mul_3:z:0add_1/y:output:0*
T0*/
_output_shapes
:??????????
	truediv_1RealDiv7div_no_nan_rank_similarity_distance_based_minkowski_pow	add_1:z:0*
T0*/
_output_shapes
:?????????e
mul_4Mulresult_grads_0truediv_1:z:0*
T0*/
_output_shapes
:?????????P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
	truediv_2RealDivtruediv_2/x:output:07sub_rank_similarity_distance_based_minkowski_expanddims*
T0*/
_output_shapes
:??????????
div_no_nan_1DivNoNan4pow_1_rank_similarity_distance_based_minkowski_pow_19div_no_nan_1_rank_similarity_distance_based_minkowski_sum*
T0*/
_output_shapes
:?????????g
mul_5Multruediv_2:z:0div_no_nan_1:z:0*
T0*/
_output_shapes
:?????????L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
add_2AddV20pow_rank_similarity_distance_based_minkowski_absadd_2/y:output:0*
T0*/
_output_shapes
:?????????O
LogLog	add_2:z:0*
T0*/
_output_shapes
:??????????
mul_6Mul4mul_6_rank_similarity_distance_based_minkowski_mul_1Log:y:0*
T0*/
_output_shapes
:?????????`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
SumSum	mul_6:z:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(_
mul_7Mul	mul_5:z:0Sum:output:0*
T0*/
_output_shapes
:?????????L
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
pow_3Pow7sub_rank_similarity_distance_based_minkowski_expanddimspow_3/y:output:0*
T0*/
_output_shapes
:?????????P
truediv_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??o
	truediv_3RealDivtruediv_3/x:output:0	pow_3:z:0*
T0*/
_output_shapes
:??????????
mul_8Multruediv_3:z:04pow_1_rank_similarity_distance_based_minkowski_pow_1*
T0*/
_output_shapes
:?????????L
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
add_3AddV29div_no_nan_1_rank_similarity_distance_based_minkowski_sumadd_3/y:output:0*
T0*/
_output_shapes
:?????????Q
Log_1Log	add_3:z:0*
T0*/
_output_shapes
:?????????\
mul_9Mul	mul_8:z:0	Log_1:y:0*
T0*/
_output_shapes
:?????????\
sub_2Sub	mul_7:z:0	mul_9:z:0*
T0*/
_output_shapes
:?????????b
mul_10Mulresult_grads_0	sub_2:z:0*
T0*/
_output_shapes
:?????????t
SqueezeSqueeze
mul_10:z:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????[

Identity_1Identity	mul_4:z:0*
T0*/
_output_shapes
:?????????^

Identity_2IdentitySqueeze:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:_ [
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:_[
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_2:[W
+
_output_shapes
:?????????
(
_user_specified_nameresult_grads_3:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????:5	1
/
_output_shapes
:?????????:5
1
/
_output_shapes
:?????????:51
/
_output_shapes
:?????????
?
?
0__inference_rank_similarity_layer_call_fn_119181
inputs_2rank1_stimulus_set
inputs_agent_id
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5: 
	unknown_6
	unknown_7: 
	unknown_8:
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_2rank1_stimulus_setinputs_agent_idunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:?????????: *,
_read_only_resource_inputs


*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_rank_similarity_layer_call_and_return_conditional_losses_117747s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:?????????:?????????:: : : : : : : : : : : : ::: 22
StatefulPartitionedCallStatefulPartitionedCall:g c
+
_output_shapes
:?????????
4
_user_specified_nameinputs/2rank1/stimulus_set:\X
+
_output_shapes
:?????????
)
_user_specified_nameinputs/agent_id: 

_output_shapes
::

_output_shapes
: :	

_output_shapes
: :$ 

_output_shapes

::,(
&
_output_shapes
::

_output_shapes
: 
?
?
U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_118430
rank1_stimulus_set
agent_id
rank_similarity_118394
rank_similarity_118396(
rank_similarity_118398:(
rank_similarity_118400:(
rank_similarity_118402:(
rank_similarity_118404: 
rank_similarity_118406: 
rank_similarity_118408 
rank_similarity_118410: $
rank_similarity_118412: 
rank_similarity_118414:  
rank_similarity_118416:  
rank_similarity_118418: 
rank_similarity_118420
rank_similarity_118422
rank_similarity_118424
identity

identity_1??'rank_similarity/StatefulPartitionedCall?
'rank_similarity/StatefulPartitionedCallStatefulPartitionedCallrank1_stimulus_setagent_idrank_similarity_118394rank_similarity_118396rank_similarity_118398rank_similarity_118400rank_similarity_118402rank_similarity_118404rank_similarity_118406rank_similarity_118408rank_similarity_118410rank_similarity_118412rank_similarity_118414rank_similarity_118416rank_similarity_118418rank_similarity_118420rank_similarity_118422rank_similarity_118424*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:?????????: *,
_read_only_resource_inputs


*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_rank_similarity_layer_call_and_return_conditional_losses_118157?
IdentityIdentity0rank_similarity/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????p

Identity_1Identity0rank_similarity/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: p
NoOpNoOp(^rank_similarity/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:?????????:?????????:: : : : : : : : : : : : ::: 2R
'rank_similarity/StatefulPartitionedCall'rank_similarity/StatefulPartitionedCall:` \
+
_output_shapes
:?????????
-
_user_specified_name2rank1/stimulus_set:UQ
+
_output_shapes
:?????????
"
_user_specified_name
agent_id: 

_output_shapes
::

_output_shapes
: :	

_output_shapes
: :$ 

_output_shapes

::,(
&
_output_shapes
::

_output_shapes
: 
?
?
:__inference_stochastic_behavior_model_layer_call_fn_118350
rank1_stimulus_set
agent_id
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5: 
	unknown_6
	unknown_7: 
	unknown_8:
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallrank1_stimulus_setagent_idunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:?????????: *,
_read_only_resource_inputs


*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_118275s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:?????????:?????????:: : : : : : : : : : : : ::: 22
StatefulPartitionedCallStatefulPartitionedCall:` \
+
_output_shapes
:?????????
-
_user_specified_name2rank1/stimulus_set:UQ
+
_output_shapes
:?????????
"
_user_specified_name
agent_id: 

_output_shapes
::

_output_shapes
: :	

_output_shapes
: :$ 

_output_shapes

::,(
&
_output_shapes
::

_output_shapes
: 
?
?
:__inference_stochastic_behavior_model_layer_call_fn_118554
inputs_2rank1_stimulus_set
inputs_agent_id
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5: 
	unknown_6
	unknown_7: 
	unknown_8:
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_2rank1_stimulus_setinputs_agent_idunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:?????????: *,
_read_only_resource_inputs


*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_118275s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:?????????:?????????:: : : : : : : : : : : : ::: 22
StatefulPartitionedCallStatefulPartitionedCall:g c
+
_output_shapes
:?????????
4
_user_specified_nameinputs/2rank1/stimulus_set:\X
+
_output_shapes
:?????????
)
_user_specified_nameinputs/agent_id: 

_output_shapes
::

_output_shapes
: :	

_output_shapes
: :$ 

_output_shapes

::,(
&
_output_shapes
::

_output_shapes
: <
#__inference_internal_grad_fn_119933CustomGradient-119745<
#__inference_internal_grad_fn_119994CustomGradient-119451<
#__inference_internal_grad_fn_120055CustomGradient-118094<
#__inference_internal_grad_fn_120116CustomGradient-117684<
#__inference_internal_grad_fn_120177CustomGradient-119079<
#__inference_internal_grad_fn_120238CustomGradient-118785<
#__inference_internal_grad_fn_120299CustomGradient-117382"?	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
W
2rank1/stimulus_set@
%serving_default_2rank1_stimulus_set:0?????????
A
agent_id5
serving_default_agent_id:0?????????@
output_14
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
behavior
		optimizer


signatures"
_tf_keras_model
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
trace_0
trace_1
trace_2
trace_32?
:__inference_stochastic_behavior_model_layer_call_fn_117820
:__inference_stochastic_behavior_model_layer_call_fn_118515
:__inference_stochastic_behavior_model_layer_call_fn_118554
:__inference_stochastic_behavior_model_layer_call_fn_118350?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 ztrace_0ztrace_1ztrace_2ztrace_3
?
trace_0
trace_1
 trace_2
!trace_32?
U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_118848
U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_119142
U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_118390
U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_118430?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 ztrace_0ztrace_1z trace_2z!trace_3
?
"	capture_0
#	capture_1
$	capture_7
%
capture_13
&
capture_14
'
capture_15B?
!__inference__wrapped_model_1174442rank1/stimulus_setagent_id"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z"	capture_0z#	capture_1z$	capture_7z%
capture_13z&
capture_14z'
capture_15
?
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses
.percept

/kernel
0percept_adapter
1kernel_adapter
2
_z_q_shape
3
_z_r_shape"
_tf_keras_layer
?
4iter

5beta_1

6beta_2
	7decay
8learning_ratem?m?m?v?v?v?"
	optimizer
,
9serving_default"
signature_map
+:)2embedding_normal_diag/loc
;:92)embedding_normal_diag/untransformed_scale
=:;2+embedding_normal_diag_1/untransformed_scale
: 2	kl_anneal
+:)2embedding_normal_diag_1/loc
: 2minkowski/rho
P:N2Dstochastic_behavior_model/rank_similarity/distance_based/minkowski/w
":  2exponential_similarity/tau
$:" 2exponential_similarity/gamma
#:! 2exponential_similarity/beta
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
'
0"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
"	capture_0
#	capture_1
$	capture_7
%
capture_13
&
capture_14
'
capture_15B?
:__inference_stochastic_behavior_model_layer_call_fn_1178202rank1/stimulus_setagent_id"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z"	capture_0z#	capture_1z$	capture_7z%
capture_13z&
capture_14z'
capture_15
?
"	capture_0
#	capture_1
$	capture_7
%
capture_13
&
capture_14
'
capture_15B?
:__inference_stochastic_behavior_model_layer_call_fn_118515inputs/2rank1/stimulus_setinputs/agent_id"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z"	capture_0z#	capture_1z$	capture_7z%
capture_13z&
capture_14z'
capture_15
?
"	capture_0
#	capture_1
$	capture_7
%
capture_13
&
capture_14
'
capture_15B?
:__inference_stochastic_behavior_model_layer_call_fn_118554inputs/2rank1/stimulus_setinputs/agent_id"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z"	capture_0z#	capture_1z$	capture_7z%
capture_13z&
capture_14z'
capture_15
?
"	capture_0
#	capture_1
$	capture_7
%
capture_13
&
capture_14
'
capture_15B?
:__inference_stochastic_behavior_model_layer_call_fn_1183502rank1/stimulus_setagent_id"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z"	capture_0z#	capture_1z$	capture_7z%
capture_13z&
capture_14z'
capture_15
?
"	capture_0
#	capture_1
$	capture_7
%
capture_13
&
capture_14
'
capture_15B?
U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_118848inputs/2rank1/stimulus_setinputs/agent_id"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z"	capture_0z#	capture_1z$	capture_7z%
capture_13z&
capture_14z'
capture_15
?
"	capture_0
#	capture_1
$	capture_7
%
capture_13
&
capture_14
'
capture_15B?
U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_119142inputs/2rank1/stimulus_setinputs/agent_id"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z"	capture_0z#	capture_1z$	capture_7z%
capture_13z&
capture_14z'
capture_15
?
"	capture_0
#	capture_1
$	capture_7
%
capture_13
&
capture_14
'
capture_15B?
U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_1183902rank1/stimulus_setagent_id"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z"	capture_0z#	capture_1z$	capture_7z%
capture_13z&
capture_14z'
capture_15
?
"	capture_0
#	capture_1
$	capture_7
%
capture_13
&
capture_14
'
capture_15B?
U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_1184302rank1/stimulus_setagent_id"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z"	capture_0z#	capture_1z$	capture_7z%
capture_13z&
capture_14z'
capture_15
!J	
Const_5jtf.TrackableConstant
!J	
Const_4jtf.TrackableConstant
!J	
Const_3jtf.TrackableConstant
!J	
Const_2jtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstant
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
?
Atrace_0
Btrace_12?
0__inference_rank_similarity_layer_call_fn_119181
0__inference_rank_similarity_layer_call_fn_119220?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zAtrace_0zBtrace_1
?
Ctrace_0
Dtrace_12?
K__inference_rank_similarity_layer_call_and_return_conditional_losses_119514
K__inference_rank_similarity_layer_call_and_return_conditional_losses_119808?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zCtrace_0zDtrace_1
?
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses
K	posterior
	Lprior
	kl_anneal"
_tf_keras_layer
?
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses
Sdistance
T
similarity"
_tf_keras_layer
?
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses
[	_all_keys
\_input_keys
]gating_keys"
_tf_keras_layer
?
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses
d	_all_keys
e_input_keys
fgating_keys"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?
"	capture_0
#	capture_1
$	capture_7
%
capture_13
&
capture_14
'
capture_15B?
$__inference_signature_wrapper_1184762rank1/stimulus_setagent_id"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z"	capture_0z#	capture_1z$	capture_7z%
capture_13z&
capture_14z'
capture_15
N
g	variables
h	keras_api
	itotal
	jcount"
_tf_keras_metric
^
k	variables
l	keras_api
	mtotal
	ncount
o
_fn_kwargs"
_tf_keras_metric
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
<
.0
/1
02
13"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
"	capture_0
#	capture_1
$	capture_7
%
capture_13
&
capture_14
'
capture_15B?
0__inference_rank_similarity_layer_call_fn_119181inputs/2rank1/stimulus_setinputs/agent_id"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z"	capture_0z#	capture_1z$	capture_7z%
capture_13z&
capture_14z'
capture_15
?
"	capture_0
#	capture_1
$	capture_7
%
capture_13
&
capture_14
'
capture_15B?
0__inference_rank_similarity_layer_call_fn_119220inputs/2rank1/stimulus_setinputs/agent_id"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z"	capture_0z#	capture_1z$	capture_7z%
capture_13z&
capture_14z'
capture_15
?
"	capture_0
#	capture_1
$	capture_7
%
capture_13
&
capture_14
'
capture_15B?
K__inference_rank_similarity_layer_call_and_return_conditional_losses_119514inputs/2rank1/stimulus_setinputs/agent_id"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z"	capture_0z#	capture_1z$	capture_7z%
capture_13z&
capture_14z'
capture_15
?
"	capture_0
#	capture_1
$	capture_7
%
capture_13
&
capture_14
'
capture_15B?
K__inference_rank_similarity_layer_call_and_return_conditional_losses_119808inputs/2rank1/stimulus_setinputs/agent_id"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z"	capture_0z#	capture_1z$	capture_7z%
capture_13z&
capture_14z'
capture_15
C
0
1
2
3
4"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses
loc
untransformed_scale
{
embeddings"
_tf_keras_layer
?
|	variables
}trainable_variables
~regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?
_embedding"
_tf_keras_layer
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
rho
w"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
tau
	gamma
beta"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
-
g	variables"
_generic_user_object
:  (2total
:  (2count
.
m0
n1"
trackable_list_wrapper
-
k	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
G
?_distribution
?_graph_parents"
_generic_user_object
.
0
1"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
|	variables
}trainable_variables
~regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
loc
untransformed_scale
?
embeddings"
_tf_keras_layer
C
0
1
2
3
4"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
_loc
?_scale
?_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
G
?_distribution
?_graph_parents"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
9
_pretransformed_input"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
_loc
?_scale
?_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
9
_pretransformed_input"
_generic_user_object
 "
trackable_list_wrapper
0:.2 Adam/embedding_normal_diag/loc/m
@:>20Adam/embedding_normal_diag/untransformed_scale/m
B:@22Adam/embedding_normal_diag_1/untransformed_scale/m
0:.2 Adam/embedding_normal_diag/loc/v
@:>20Adam/embedding_normal_diag/untransformed_scale/v
B:@22Adam/embedding_normal_diag_1/untransformed_scale/v
wbu
&distance_based/minkowski/BroadcastTo:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_119808
obm
distance_based/minkowski/sub:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_119808
obm
distance_based/minkowski/Abs:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_119808
obm
distance_based/minkowski/Pow:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_119808
vbt
%distance_based/minkowski/ExpandDims:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_119808
qbo
 distance_based/minkowski/Pow_1:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_119808
obm
distance_based/minkowski/Sum:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_119808
qbo
 distance_based/minkowski/Mul_1:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_119808
wbu
&distance_based/minkowski/BroadcastTo:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_119514
obm
distance_based/minkowski/sub:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_119514
obm
distance_based/minkowski/Abs:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_119514
obm
distance_based/minkowski/Pow:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_119514
vbt
%distance_based/minkowski/ExpandDims:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_119514
qbo
 distance_based/minkowski/Pow_1:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_119514
obm
distance_based/minkowski/Sum:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_119514
qbo
 distance_based/minkowski/Mul_1:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_119514
wbu
&distance_based/minkowski/BroadcastTo:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_118157
obm
distance_based/minkowski/sub:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_118157
obm
distance_based/minkowski/Abs:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_118157
obm
distance_based/minkowski/Pow:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_118157
vbt
%distance_based/minkowski/ExpandDims:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_118157
qbo
 distance_based/minkowski/Pow_1:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_118157
obm
distance_based/minkowski/Sum:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_118157
qbo
 distance_based/minkowski/Mul_1:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_118157
wbu
&distance_based/minkowski/BroadcastTo:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_117747
obm
distance_based/minkowski/sub:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_117747
obm
distance_based/minkowski/Abs:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_117747
obm
distance_based/minkowski/Pow:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_117747
vbt
%distance_based/minkowski/ExpandDims:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_117747
qbo
 distance_based/minkowski/Pow_1:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_117747
obm
distance_based/minkowski/Sum:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_117747
qbo
 distance_based/minkowski/Mul_1:0K__inference_rank_similarity_layer_call_and_return_conditional_losses_117747
?b?
6rank_similarity/distance_based/minkowski/BroadcastTo:0U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_119142
?b?
.rank_similarity/distance_based/minkowski/sub:0U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_119142
?b?
.rank_similarity/distance_based/minkowski/Abs:0U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_119142
?b?
.rank_similarity/distance_based/minkowski/Pow:0U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_119142
?b?
5rank_similarity/distance_based/minkowski/ExpandDims:0U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_119142
?b?
0rank_similarity/distance_based/minkowski/Pow_1:0U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_119142
?b?
.rank_similarity/distance_based/minkowski/Sum:0U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_119142
?b?
0rank_similarity/distance_based/minkowski/Mul_1:0U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_119142
?b?
6rank_similarity/distance_based/minkowski/BroadcastTo:0U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_118848
?b?
.rank_similarity/distance_based/minkowski/sub:0U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_118848
?b?
.rank_similarity/distance_based/minkowski/Abs:0U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_118848
?b?
.rank_similarity/distance_based/minkowski/Pow:0U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_118848
?b?
5rank_similarity/distance_based/minkowski/ExpandDims:0U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_118848
?b?
0rank_similarity/distance_based/minkowski/Pow_1:0U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_118848
?b?
.rank_similarity/distance_based/minkowski/Sum:0U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_118848
?b?
0rank_similarity/distance_based/minkowski/Mul_1:0U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_118848
wbu
Pstochastic_behavior_model/rank_similarity/distance_based/minkowski/BroadcastTo:0!__inference__wrapped_model_117444
obm
Hstochastic_behavior_model/rank_similarity/distance_based/minkowski/sub:0!__inference__wrapped_model_117444
obm
Hstochastic_behavior_model/rank_similarity/distance_based/minkowski/Abs:0!__inference__wrapped_model_117444
obm
Hstochastic_behavior_model/rank_similarity/distance_based/minkowski/Pow:0!__inference__wrapped_model_117444
vbt
Ostochastic_behavior_model/rank_similarity/distance_based/minkowski/ExpandDims:0!__inference__wrapped_model_117444
qbo
Jstochastic_behavior_model/rank_similarity/distance_based/minkowski/Pow_1:0!__inference__wrapped_model_117444
obm
Hstochastic_behavior_model/rank_similarity/distance_based/minkowski/Sum:0!__inference__wrapped_model_117444
qbo
Jstochastic_behavior_model/rank_similarity/distance_based/minkowski/Mul_1:0!__inference__wrapped_model_117444?
!__inference__wrapped_model_117444?"#$%&'???
???
??~
H
2rank1/stimulus_set1?.
2rank1/stimulus_set?????????
2
agent_id&?#
agent_id?????????
? "7?4
2
output_1&?#
output_1??????????
#__inference_internal_grad_fn_119933????????????
???

 
0?-
result_grads_0?????????
0?-
result_grads_1?????????
0?-
result_grads_2?????????
,?)
result_grads_3?????????
? "r?o

 
#? 
1?????????
#? 
2?????????
?
3??????????
#__inference_internal_grad_fn_119994????????????
???

 
0?-
result_grads_0?????????
0?-
result_grads_1?????????
0?-
result_grads_2?????????
,?)
result_grads_3?????????
? "r?o

 
#? 
1?????????
#? 
2?????????
?
3??????????
#__inference_internal_grad_fn_120055????????????
???

 
0?-
result_grads_0?????????
0?-
result_grads_1?????????
0?-
result_grads_2?????????
,?)
result_grads_3?????????
? "r?o

 
#? 
1?????????
#? 
2?????????
?
3??????????
#__inference_internal_grad_fn_120116????????????
???

 
0?-
result_grads_0?????????
0?-
result_grads_1?????????
0?-
result_grads_2?????????
,?)
result_grads_3?????????
? "r?o

 
#? 
1?????????
#? 
2?????????
?
3??????????
#__inference_internal_grad_fn_120177????????????
???

 
0?-
result_grads_0?????????
0?-
result_grads_1?????????
0?-
result_grads_2?????????
,?)
result_grads_3?????????
? "r?o

 
#? 
1?????????
#? 
2?????????
?
3??????????
#__inference_internal_grad_fn_120238????????????
???

 
0?-
result_grads_0?????????
0?-
result_grads_1?????????
0?-
result_grads_2?????????
,?)
result_grads_3?????????
? "r?o

 
#? 
1?????????
#? 
2?????????
?
3??????????
#__inference_internal_grad_fn_120299????????????
???

 
0?-
result_grads_0?????????
0?-
result_grads_1?????????
0?-
result_grads_2?????????
,?)
result_grads_3?????????
? "r?o

 
#? 
1?????????
#? 
2?????????
?
3??????????
K__inference_rank_similarity_layer_call_and_return_conditional_losses_119514?"#$%&'???
???
???
O
2rank1/stimulus_set8?5
inputs/2rank1/stimulus_set?????????
9
agent_id-?*
inputs/agent_id?????????
p 
? "7?4
?
0?????????
?
?	
1/0 ?
K__inference_rank_similarity_layer_call_and_return_conditional_losses_119808?"#$%&'???
???
???
O
2rank1/stimulus_set8?5
inputs/2rank1/stimulus_set?????????
9
agent_id-?*
inputs/agent_id?????????
p
? "7?4
?
0?????????
?
?	
1/0 ?
0__inference_rank_similarity_layer_call_fn_119181?"#$%&'???
???
???
O
2rank1/stimulus_set8?5
inputs/2rank1/stimulus_set?????????
9
agent_id-?*
inputs/agent_id?????????
p 
? "???????????
0__inference_rank_similarity_layer_call_fn_119220?"#$%&'???
???
???
O
2rank1/stimulus_set8?5
inputs/2rank1/stimulus_set?????????
9
agent_id-?*
inputs/agent_id?????????
p
? "???????????
$__inference_signature_wrapper_118476?"#$%&'???
? 
??~
H
2rank1/stimulus_set1?.
2rank1/stimulus_set?????????
2
agent_id&?#
agent_id?????????"7?4
2
output_1&?#
output_1??????????
U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_118390?"#$%&'???
???
??~
H
2rank1/stimulus_set1?.
2rank1/stimulus_set?????????
2
agent_id&?#
agent_id?????????
?

trainingp "7?4
?
0?????????
?
?	
1/0 ?
U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_118430?"#$%&'???
???
??~
H
2rank1/stimulus_set1?.
2rank1/stimulus_set?????????
2
agent_id&?#
agent_id?????????
?

trainingp"7?4
?
0?????????
?
?	
1/0 ?
U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_118848?"#$%&'???
???
???
O
2rank1/stimulus_set8?5
inputs/2rank1/stimulus_set?????????
9
agent_id-?*
inputs/agent_id?????????
?

trainingp "7?4
?
0?????????
?
?	
1/0 ?
U__inference_stochastic_behavior_model_layer_call_and_return_conditional_losses_119142?"#$%&'???
???
???
O
2rank1/stimulus_set8?5
inputs/2rank1/stimulus_set?????????
9
agent_id-?*
inputs/agent_id?????????
?

trainingp"7?4
?
0?????????
?
?	
1/0 ?
:__inference_stochastic_behavior_model_layer_call_fn_117820?"#$%&'???
???
??~
H
2rank1/stimulus_set1?.
2rank1/stimulus_set?????????
2
agent_id&?#
agent_id?????????
?

trainingp "???????????
:__inference_stochastic_behavior_model_layer_call_fn_118350?"#$%&'???
???
??~
H
2rank1/stimulus_set1?.
2rank1/stimulus_set?????????
2
agent_id&?#
agent_id?????????
?

trainingp"???????????
:__inference_stochastic_behavior_model_layer_call_fn_118515?"#$%&'???
???
???
O
2rank1/stimulus_set8?5
inputs/2rank1/stimulus_set?????????
9
agent_id-?*
inputs/agent_id?????????
?

trainingp "???????????
:__inference_stochastic_behavior_model_layer_call_fn_118554?"#$%&'???
???
???
O
2rank1/stimulus_set8?5
inputs/2rank1/stimulus_set?????????
9
agent_id-?*
inputs/agent_id?????????
?

trainingp"??????????