??
?8?8
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
?
ArgMax

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"
output_typetype0	:
2	
E
AssignAddVariableOp
resource
value"dtype"
dtypetype?
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
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
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
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
A
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
?
FloorMod
x"T
y"T
z"T"
Ttype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
$

LogicalAnd
x

y

z
?
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
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
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
?
Multinomial
logits"T
num_samples
output"output_dtype"
seedint "
seed2int "
Ttype:
2	" 
output_dtypetype0	:
2	?
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
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
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
?
RandomUniformInt

shape"T
minval"Tout
maxval"Tout
output"Tout"
seedint "
seed2int "
Touttype:
2	"
Ttype:
2	?
e
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2		
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
7

Reciprocal
x"T
y"T"
Ttype:
2
	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceApplyAdam
var
m
v
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( ?
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
A
SelectV2
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
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
#SparseSoftmaxCrossEntropyWithLogits
features"T
labels"Tlabels	
loss"T
backprop"T"
Ttype:
2"
Tlabelstype0	:
2	
3
Square
x"T
y"T"
Ttype:
2
	
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
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
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
9
VarIsInitializedOp
resource
is_initialized
?
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.7.02v2.7.0-rc1-69-gc256c071bb2??
?
default_policy/observationsPlaceholder*/
_output_shapes
:?????????TT*
dtype0*$
shape:?????????TT
?
<default_policy/conv1/kernel/Initializer/random_uniform/shapeConst*.
_class$
" loc:@default_policy/conv1/kernel*
_output_shapes
:*
dtype0*%
valueB"            
?
:default_policy/conv1/kernel/Initializer/random_uniform/minConst*.
_class$
" loc:@default_policy/conv1/kernel*
_output_shapes
: *
dtype0*
valueB
 *?7??
?
:default_policy/conv1/kernel/Initializer/random_uniform/maxConst*.
_class$
" loc:@default_policy/conv1/kernel*
_output_shapes
: *
dtype0*
valueB
 *?7?=
?
Ddefault_policy/conv1/kernel/Initializer/random_uniform/RandomUniformRandomUniform<default_policy/conv1/kernel/Initializer/random_uniform/shape*
T0*.
_class$
" loc:@default_policy/conv1/kernel*&
_output_shapes
:*
dtype0*

seed *
seed2 
?
:default_policy/conv1/kernel/Initializer/random_uniform/subSub:default_policy/conv1/kernel/Initializer/random_uniform/max:default_policy/conv1/kernel/Initializer/random_uniform/min*
T0*.
_class$
" loc:@default_policy/conv1/kernel*
_output_shapes
: 
?
:default_policy/conv1/kernel/Initializer/random_uniform/mulMulDdefault_policy/conv1/kernel/Initializer/random_uniform/RandomUniform:default_policy/conv1/kernel/Initializer/random_uniform/sub*
T0*.
_class$
" loc:@default_policy/conv1/kernel*&
_output_shapes
:
?
6default_policy/conv1/kernel/Initializer/random_uniformAddV2:default_policy/conv1/kernel/Initializer/random_uniform/mul:default_policy/conv1/kernel/Initializer/random_uniform/min*
T0*.
_class$
" loc:@default_policy/conv1/kernel*&
_output_shapes
:
?
default_policy/conv1/kernelVarHandleOp*.
_class$
" loc:@default_policy/conv1/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*,
shared_namedefault_policy/conv1/kernel
?
<default_policy/conv1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdefault_policy/conv1/kernel*
_output_shapes
: 
?
"default_policy/conv1/kernel/AssignAssignVariableOpdefault_policy/conv1/kernel6default_policy/conv1/kernel/Initializer/random_uniform*
dtype0
?
/default_policy/conv1/kernel/Read/ReadVariableOpReadVariableOpdefault_policy/conv1/kernel*&
_output_shapes
:*
dtype0
?
+default_policy/conv1/bias/Initializer/zerosConst*,
_class"
 loc:@default_policy/conv1/bias*
_output_shapes
:*
dtype0*
valueB*    
?
default_policy/conv1/biasVarHandleOp*,
_class"
 loc:@default_policy/conv1/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:**
shared_namedefault_policy/conv1/bias
?
:default_policy/conv1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdefault_policy/conv1/bias*
_output_shapes
: 
?
 default_policy/conv1/bias/AssignAssignVariableOpdefault_policy/conv1/bias+default_policy/conv1/bias/Initializer/zeros*
dtype0
?
-default_policy/conv1/bias/Read/ReadVariableOpReadVariableOpdefault_policy/conv1/bias*
_output_shapes
:*
dtype0
?
*default_policy/conv1/Conv2D/ReadVariableOpReadVariableOpdefault_policy/conv1/kernel*&
_output_shapes
:*
dtype0
?
default_policy/conv1/Conv2DConv2Ddefault_policy/observations*default_policy/conv1/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
+default_policy/conv1/BiasAdd/ReadVariableOpReadVariableOpdefault_policy/conv1/bias*
_output_shapes
:*
dtype0
?
default_policy/conv1/BiasAddBiasAdddefault_policy/conv1/Conv2D+default_policy/conv1/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:?????????*
data_formatNHWC
y
default_policy/conv1/ReluReludefault_policy/conv1/BiasAdd*
T0*/
_output_shapes
:?????????
?
<default_policy/conv2/kernel/Initializer/random_uniform/shapeConst*.
_class$
" loc:@default_policy/conv2/kernel*
_output_shapes
:*
dtype0*%
valueB"             
?
:default_policy/conv2/kernel/Initializer/random_uniform/minConst*.
_class$
" loc:@default_policy/conv2/kernel*
_output_shapes
: *
dtype0*
valueB
 *???
?
:default_policy/conv2/kernel/Initializer/random_uniform/maxConst*.
_class$
" loc:@default_policy/conv2/kernel*
_output_shapes
: *
dtype0*
valueB
 *??=
?
Ddefault_policy/conv2/kernel/Initializer/random_uniform/RandomUniformRandomUniform<default_policy/conv2/kernel/Initializer/random_uniform/shape*
T0*.
_class$
" loc:@default_policy/conv2/kernel*&
_output_shapes
: *
dtype0*

seed *
seed2 
?
:default_policy/conv2/kernel/Initializer/random_uniform/subSub:default_policy/conv2/kernel/Initializer/random_uniform/max:default_policy/conv2/kernel/Initializer/random_uniform/min*
T0*.
_class$
" loc:@default_policy/conv2/kernel*
_output_shapes
: 
?
:default_policy/conv2/kernel/Initializer/random_uniform/mulMulDdefault_policy/conv2/kernel/Initializer/random_uniform/RandomUniform:default_policy/conv2/kernel/Initializer/random_uniform/sub*
T0*.
_class$
" loc:@default_policy/conv2/kernel*&
_output_shapes
: 
?
6default_policy/conv2/kernel/Initializer/random_uniformAddV2:default_policy/conv2/kernel/Initializer/random_uniform/mul:default_policy/conv2/kernel/Initializer/random_uniform/min*
T0*.
_class$
" loc:@default_policy/conv2/kernel*&
_output_shapes
: 
?
default_policy/conv2/kernelVarHandleOp*.
_class$
" loc:@default_policy/conv2/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *,
shared_namedefault_policy/conv2/kernel
?
<default_policy/conv2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdefault_policy/conv2/kernel*
_output_shapes
: 
?
"default_policy/conv2/kernel/AssignAssignVariableOpdefault_policy/conv2/kernel6default_policy/conv2/kernel/Initializer/random_uniform*
dtype0
?
/default_policy/conv2/kernel/Read/ReadVariableOpReadVariableOpdefault_policy/conv2/kernel*&
_output_shapes
: *
dtype0
?
+default_policy/conv2/bias/Initializer/zerosConst*,
_class"
 loc:@default_policy/conv2/bias*
_output_shapes
: *
dtype0*
valueB *    
?
default_policy/conv2/biasVarHandleOp*,
_class"
 loc:@default_policy/conv2/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: **
shared_namedefault_policy/conv2/bias
?
:default_policy/conv2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdefault_policy/conv2/bias*
_output_shapes
: 
?
 default_policy/conv2/bias/AssignAssignVariableOpdefault_policy/conv2/bias+default_policy/conv2/bias/Initializer/zeros*
dtype0
?
-default_policy/conv2/bias/Read/ReadVariableOpReadVariableOpdefault_policy/conv2/bias*
_output_shapes
: *
dtype0
?
*default_policy/conv2/Conv2D/ReadVariableOpReadVariableOpdefault_policy/conv2/kernel*&
_output_shapes
: *
dtype0
?
default_policy/conv2/Conv2DConv2Ddefault_policy/conv1/Relu*default_policy/conv2/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? *
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
+default_policy/conv2/BiasAdd/ReadVariableOpReadVariableOpdefault_policy/conv2/bias*
_output_shapes
: *
dtype0
?
default_policy/conv2/BiasAddBiasAdddefault_policy/conv2/Conv2D+default_policy/conv2/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:????????? *
data_formatNHWC
y
default_policy/conv2/ReluReludefault_policy/conv2/BiasAdd*
T0*/
_output_shapes
:????????? 
?
<default_policy/conv3/kernel/Initializer/random_uniform/shapeConst*.
_class$
" loc:@default_policy/conv3/kernel*
_output_shapes
:*
dtype0*%
valueB"             
?
:default_policy/conv3/kernel/Initializer/random_uniform/minConst*.
_class$
" loc:@default_policy/conv3/kernel*
_output_shapes
: *
dtype0*
valueB
 *??V?
?
:default_policy/conv3/kernel/Initializer/random_uniform/maxConst*.
_class$
" loc:@default_policy/conv3/kernel*
_output_shapes
: *
dtype0*
valueB
 *??V<
?
Ddefault_policy/conv3/kernel/Initializer/random_uniform/RandomUniformRandomUniform<default_policy/conv3/kernel/Initializer/random_uniform/shape*
T0*.
_class$
" loc:@default_policy/conv3/kernel*'
_output_shapes
: ?*
dtype0*

seed *
seed2 
?
:default_policy/conv3/kernel/Initializer/random_uniform/subSub:default_policy/conv3/kernel/Initializer/random_uniform/max:default_policy/conv3/kernel/Initializer/random_uniform/min*
T0*.
_class$
" loc:@default_policy/conv3/kernel*
_output_shapes
: 
?
:default_policy/conv3/kernel/Initializer/random_uniform/mulMulDdefault_policy/conv3/kernel/Initializer/random_uniform/RandomUniform:default_policy/conv3/kernel/Initializer/random_uniform/sub*
T0*.
_class$
" loc:@default_policy/conv3/kernel*'
_output_shapes
: ?
?
6default_policy/conv3/kernel/Initializer/random_uniformAddV2:default_policy/conv3/kernel/Initializer/random_uniform/mul:default_policy/conv3/kernel/Initializer/random_uniform/min*
T0*.
_class$
" loc:@default_policy/conv3/kernel*'
_output_shapes
: ?
?
default_policy/conv3/kernelVarHandleOp*.
_class$
" loc:@default_policy/conv3/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: ?*,
shared_namedefault_policy/conv3/kernel
?
<default_policy/conv3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdefault_policy/conv3/kernel*
_output_shapes
: 
?
"default_policy/conv3/kernel/AssignAssignVariableOpdefault_policy/conv3/kernel6default_policy/conv3/kernel/Initializer/random_uniform*
dtype0
?
/default_policy/conv3/kernel/Read/ReadVariableOpReadVariableOpdefault_policy/conv3/kernel*'
_output_shapes
: ?*
dtype0
?
+default_policy/conv3/bias/Initializer/zerosConst*,
_class"
 loc:@default_policy/conv3/bias*
_output_shapes	
:?*
dtype0*
valueB?*    
?
default_policy/conv3/biasVarHandleOp*,
_class"
 loc:@default_policy/conv3/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:?**
shared_namedefault_policy/conv3/bias
?
:default_policy/conv3/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdefault_policy/conv3/bias*
_output_shapes
: 
?
 default_policy/conv3/bias/AssignAssignVariableOpdefault_policy/conv3/bias+default_policy/conv3/bias/Initializer/zeros*
dtype0
?
-default_policy/conv3/bias/Read/ReadVariableOpReadVariableOpdefault_policy/conv3/bias*
_output_shapes	
:?*
dtype0
?
*default_policy/conv3/Conv2D/ReadVariableOpReadVariableOpdefault_policy/conv3/kernel*'
_output_shapes
: ?*
dtype0
?
default_policy/conv3/Conv2DConv2Ddefault_policy/conv2/Relu*default_policy/conv3/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingVALID*
strides
*
use_cudnn_on_gpu(
?
+default_policy/conv3/BiasAdd/ReadVariableOpReadVariableOpdefault_policy/conv3/bias*
_output_shapes	
:?*
dtype0
?
default_policy/conv3/BiasAddBiasAdddefault_policy/conv3/Conv2D+default_policy/conv3/BiasAdd/ReadVariableOp*
T0*0
_output_shapes
:??????????*
data_formatNHWC
z
default_policy/conv3/ReluReludefault_policy/conv3/BiasAdd*
T0*0
_output_shapes
:??????????
?
?default_policy/conv_out/kernel/Initializer/random_uniform/shapeConst*1
_class'
%#loc:@default_policy/conv_out/kernel*
_output_shapes
:*
dtype0*%
valueB"            
?
=default_policy/conv_out/kernel/Initializer/random_uniform/minConst*1
_class'
%#loc:@default_policy/conv_out/kernel*
_output_shapes
: *
dtype0*
valueB
 *???
?
=default_policy/conv_out/kernel/Initializer/random_uniform/maxConst*1
_class'
%#loc:@default_policy/conv_out/kernel*
_output_shapes
: *
dtype0*
valueB
 *??>
?
Gdefault_policy/conv_out/kernel/Initializer/random_uniform/RandomUniformRandomUniform?default_policy/conv_out/kernel/Initializer/random_uniform/shape*
T0*1
_class'
%#loc:@default_policy/conv_out/kernel*'
_output_shapes
:?*
dtype0*

seed *
seed2 
?
=default_policy/conv_out/kernel/Initializer/random_uniform/subSub=default_policy/conv_out/kernel/Initializer/random_uniform/max=default_policy/conv_out/kernel/Initializer/random_uniform/min*
T0*1
_class'
%#loc:@default_policy/conv_out/kernel*
_output_shapes
: 
?
=default_policy/conv_out/kernel/Initializer/random_uniform/mulMulGdefault_policy/conv_out/kernel/Initializer/random_uniform/RandomUniform=default_policy/conv_out/kernel/Initializer/random_uniform/sub*
T0*1
_class'
%#loc:@default_policy/conv_out/kernel*'
_output_shapes
:?
?
9default_policy/conv_out/kernel/Initializer/random_uniformAddV2=default_policy/conv_out/kernel/Initializer/random_uniform/mul=default_policy/conv_out/kernel/Initializer/random_uniform/min*
T0*1
_class'
%#loc:@default_policy/conv_out/kernel*'
_output_shapes
:?
?
default_policy/conv_out/kernelVarHandleOp*1
_class'
%#loc:@default_policy/conv_out/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:?*/
shared_name default_policy/conv_out/kernel
?
?default_policy/conv_out/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdefault_policy/conv_out/kernel*
_output_shapes
: 
?
%default_policy/conv_out/kernel/AssignAssignVariableOpdefault_policy/conv_out/kernel9default_policy/conv_out/kernel/Initializer/random_uniform*
dtype0
?
2default_policy/conv_out/kernel/Read/ReadVariableOpReadVariableOpdefault_policy/conv_out/kernel*'
_output_shapes
:?*
dtype0
?
.default_policy/conv_out/bias/Initializer/zerosConst*/
_class%
#!loc:@default_policy/conv_out/bias*
_output_shapes
:*
dtype0*
valueB*    
?
default_policy/conv_out/biasVarHandleOp*/
_class%
#!loc:@default_policy/conv_out/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*-
shared_namedefault_policy/conv_out/bias
?
=default_policy/conv_out/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdefault_policy/conv_out/bias*
_output_shapes
: 
?
#default_policy/conv_out/bias/AssignAssignVariableOpdefault_policy/conv_out/bias.default_policy/conv_out/bias/Initializer/zeros*
dtype0
?
0default_policy/conv_out/bias/Read/ReadVariableOpReadVariableOpdefault_policy/conv_out/bias*
_output_shapes
:*
dtype0
?
-default_policy/conv_out/Conv2D/ReadVariableOpReadVariableOpdefault_policy/conv_out/kernel*'
_output_shapes
:?*
dtype0
?
default_policy/conv_out/Conv2DConv2Ddefault_policy/conv3/Relu-default_policy/conv_out/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
.default_policy/conv_out/BiasAdd/ReadVariableOpReadVariableOpdefault_policy/conv_out/bias*
_output_shapes
:*
dtype0
?
default_policy/conv_out/BiasAddBiasAdddefault_policy/conv_out/Conv2D.default_policy/conv_out/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:?????????*
data_formatNHWC
?
Cdefault_policy/conv_value_1/kernel/Initializer/random_uniform/shapeConst*5
_class+
)'loc:@default_policy/conv_value_1/kernel*
_output_shapes
:*
dtype0*%
valueB"            
?
Adefault_policy/conv_value_1/kernel/Initializer/random_uniform/minConst*5
_class+
)'loc:@default_policy/conv_value_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *?7??
?
Adefault_policy/conv_value_1/kernel/Initializer/random_uniform/maxConst*5
_class+
)'loc:@default_policy/conv_value_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *?7?=
?
Kdefault_policy/conv_value_1/kernel/Initializer/random_uniform/RandomUniformRandomUniformCdefault_policy/conv_value_1/kernel/Initializer/random_uniform/shape*
T0*5
_class+
)'loc:@default_policy/conv_value_1/kernel*&
_output_shapes
:*
dtype0*

seed *
seed2 
?
Adefault_policy/conv_value_1/kernel/Initializer/random_uniform/subSubAdefault_policy/conv_value_1/kernel/Initializer/random_uniform/maxAdefault_policy/conv_value_1/kernel/Initializer/random_uniform/min*
T0*5
_class+
)'loc:@default_policy/conv_value_1/kernel*
_output_shapes
: 
?
Adefault_policy/conv_value_1/kernel/Initializer/random_uniform/mulMulKdefault_policy/conv_value_1/kernel/Initializer/random_uniform/RandomUniformAdefault_policy/conv_value_1/kernel/Initializer/random_uniform/sub*
T0*5
_class+
)'loc:@default_policy/conv_value_1/kernel*&
_output_shapes
:
?
=default_policy/conv_value_1/kernel/Initializer/random_uniformAddV2Adefault_policy/conv_value_1/kernel/Initializer/random_uniform/mulAdefault_policy/conv_value_1/kernel/Initializer/random_uniform/min*
T0*5
_class+
)'loc:@default_policy/conv_value_1/kernel*&
_output_shapes
:
?
"default_policy/conv_value_1/kernelVarHandleOp*5
_class+
)'loc:@default_policy/conv_value_1/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*3
shared_name$"default_policy/conv_value_1/kernel
?
Cdefault_policy/conv_value_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp"default_policy/conv_value_1/kernel*
_output_shapes
: 
?
)default_policy/conv_value_1/kernel/AssignAssignVariableOp"default_policy/conv_value_1/kernel=default_policy/conv_value_1/kernel/Initializer/random_uniform*
dtype0
?
6default_policy/conv_value_1/kernel/Read/ReadVariableOpReadVariableOp"default_policy/conv_value_1/kernel*&
_output_shapes
:*
dtype0
?
2default_policy/conv_value_1/bias/Initializer/zerosConst*3
_class)
'%loc:@default_policy/conv_value_1/bias*
_output_shapes
:*
dtype0*
valueB*    
?
 default_policy/conv_value_1/biasVarHandleOp*3
_class)
'%loc:@default_policy/conv_value_1/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*1
shared_name" default_policy/conv_value_1/bias
?
Adefault_policy/conv_value_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp default_policy/conv_value_1/bias*
_output_shapes
: 
?
'default_policy/conv_value_1/bias/AssignAssignVariableOp default_policy/conv_value_1/bias2default_policy/conv_value_1/bias/Initializer/zeros*
dtype0
?
4default_policy/conv_value_1/bias/Read/ReadVariableOpReadVariableOp default_policy/conv_value_1/bias*
_output_shapes
:*
dtype0
?
1default_policy/conv_value_1/Conv2D/ReadVariableOpReadVariableOp"default_policy/conv_value_1/kernel*&
_output_shapes
:*
dtype0
?
"default_policy/conv_value_1/Conv2DConv2Ddefault_policy/observations1default_policy/conv_value_1/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
2default_policy/conv_value_1/BiasAdd/ReadVariableOpReadVariableOp default_policy/conv_value_1/bias*
_output_shapes
:*
dtype0
?
#default_policy/conv_value_1/BiasAddBiasAdd"default_policy/conv_value_1/Conv2D2default_policy/conv_value_1/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:?????????*
data_formatNHWC
?
 default_policy/conv_value_1/ReluRelu#default_policy/conv_value_1/BiasAdd*
T0*/
_output_shapes
:?????????
?
Cdefault_policy/conv_value_2/kernel/Initializer/random_uniform/shapeConst*5
_class+
)'loc:@default_policy/conv_value_2/kernel*
_output_shapes
:*
dtype0*%
valueB"             
?
Adefault_policy/conv_value_2/kernel/Initializer/random_uniform/minConst*5
_class+
)'loc:@default_policy/conv_value_2/kernel*
_output_shapes
: *
dtype0*
valueB
 *???
?
Adefault_policy/conv_value_2/kernel/Initializer/random_uniform/maxConst*5
_class+
)'loc:@default_policy/conv_value_2/kernel*
_output_shapes
: *
dtype0*
valueB
 *??=
?
Kdefault_policy/conv_value_2/kernel/Initializer/random_uniform/RandomUniformRandomUniformCdefault_policy/conv_value_2/kernel/Initializer/random_uniform/shape*
T0*5
_class+
)'loc:@default_policy/conv_value_2/kernel*&
_output_shapes
: *
dtype0*

seed *
seed2 
?
Adefault_policy/conv_value_2/kernel/Initializer/random_uniform/subSubAdefault_policy/conv_value_2/kernel/Initializer/random_uniform/maxAdefault_policy/conv_value_2/kernel/Initializer/random_uniform/min*
T0*5
_class+
)'loc:@default_policy/conv_value_2/kernel*
_output_shapes
: 
?
Adefault_policy/conv_value_2/kernel/Initializer/random_uniform/mulMulKdefault_policy/conv_value_2/kernel/Initializer/random_uniform/RandomUniformAdefault_policy/conv_value_2/kernel/Initializer/random_uniform/sub*
T0*5
_class+
)'loc:@default_policy/conv_value_2/kernel*&
_output_shapes
: 
?
=default_policy/conv_value_2/kernel/Initializer/random_uniformAddV2Adefault_policy/conv_value_2/kernel/Initializer/random_uniform/mulAdefault_policy/conv_value_2/kernel/Initializer/random_uniform/min*
T0*5
_class+
)'loc:@default_policy/conv_value_2/kernel*&
_output_shapes
: 
?
"default_policy/conv_value_2/kernelVarHandleOp*5
_class+
)'loc:@default_policy/conv_value_2/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *3
shared_name$"default_policy/conv_value_2/kernel
?
Cdefault_policy/conv_value_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp"default_policy/conv_value_2/kernel*
_output_shapes
: 
?
)default_policy/conv_value_2/kernel/AssignAssignVariableOp"default_policy/conv_value_2/kernel=default_policy/conv_value_2/kernel/Initializer/random_uniform*
dtype0
?
6default_policy/conv_value_2/kernel/Read/ReadVariableOpReadVariableOp"default_policy/conv_value_2/kernel*&
_output_shapes
: *
dtype0
?
2default_policy/conv_value_2/bias/Initializer/zerosConst*3
_class)
'%loc:@default_policy/conv_value_2/bias*
_output_shapes
: *
dtype0*
valueB *    
?
 default_policy/conv_value_2/biasVarHandleOp*3
_class)
'%loc:@default_policy/conv_value_2/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *1
shared_name" default_policy/conv_value_2/bias
?
Adefault_policy/conv_value_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp default_policy/conv_value_2/bias*
_output_shapes
: 
?
'default_policy/conv_value_2/bias/AssignAssignVariableOp default_policy/conv_value_2/bias2default_policy/conv_value_2/bias/Initializer/zeros*
dtype0
?
4default_policy/conv_value_2/bias/Read/ReadVariableOpReadVariableOp default_policy/conv_value_2/bias*
_output_shapes
: *
dtype0
?
1default_policy/conv_value_2/Conv2D/ReadVariableOpReadVariableOp"default_policy/conv_value_2/kernel*&
_output_shapes
: *
dtype0
?
"default_policy/conv_value_2/Conv2DConv2D default_policy/conv_value_1/Relu1default_policy/conv_value_2/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? *
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
2default_policy/conv_value_2/BiasAdd/ReadVariableOpReadVariableOp default_policy/conv_value_2/bias*
_output_shapes
: *
dtype0
?
#default_policy/conv_value_2/BiasAddBiasAdd"default_policy/conv_value_2/Conv2D2default_policy/conv_value_2/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:????????? *
data_formatNHWC
?
 default_policy/conv_value_2/ReluRelu#default_policy/conv_value_2/BiasAdd*
T0*/
_output_shapes
:????????? 
?
Cdefault_policy/conv_value_3/kernel/Initializer/random_uniform/shapeConst*5
_class+
)'loc:@default_policy/conv_value_3/kernel*
_output_shapes
:*
dtype0*%
valueB"             
?
Adefault_policy/conv_value_3/kernel/Initializer/random_uniform/minConst*5
_class+
)'loc:@default_policy/conv_value_3/kernel*
_output_shapes
: *
dtype0*
valueB
 *??V?
?
Adefault_policy/conv_value_3/kernel/Initializer/random_uniform/maxConst*5
_class+
)'loc:@default_policy/conv_value_3/kernel*
_output_shapes
: *
dtype0*
valueB
 *??V<
?
Kdefault_policy/conv_value_3/kernel/Initializer/random_uniform/RandomUniformRandomUniformCdefault_policy/conv_value_3/kernel/Initializer/random_uniform/shape*
T0*5
_class+
)'loc:@default_policy/conv_value_3/kernel*'
_output_shapes
: ?*
dtype0*

seed *
seed2 
?
Adefault_policy/conv_value_3/kernel/Initializer/random_uniform/subSubAdefault_policy/conv_value_3/kernel/Initializer/random_uniform/maxAdefault_policy/conv_value_3/kernel/Initializer/random_uniform/min*
T0*5
_class+
)'loc:@default_policy/conv_value_3/kernel*
_output_shapes
: 
?
Adefault_policy/conv_value_3/kernel/Initializer/random_uniform/mulMulKdefault_policy/conv_value_3/kernel/Initializer/random_uniform/RandomUniformAdefault_policy/conv_value_3/kernel/Initializer/random_uniform/sub*
T0*5
_class+
)'loc:@default_policy/conv_value_3/kernel*'
_output_shapes
: ?
?
=default_policy/conv_value_3/kernel/Initializer/random_uniformAddV2Adefault_policy/conv_value_3/kernel/Initializer/random_uniform/mulAdefault_policy/conv_value_3/kernel/Initializer/random_uniform/min*
T0*5
_class+
)'loc:@default_policy/conv_value_3/kernel*'
_output_shapes
: ?
?
"default_policy/conv_value_3/kernelVarHandleOp*5
_class+
)'loc:@default_policy/conv_value_3/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: ?*3
shared_name$"default_policy/conv_value_3/kernel
?
Cdefault_policy/conv_value_3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp"default_policy/conv_value_3/kernel*
_output_shapes
: 
?
)default_policy/conv_value_3/kernel/AssignAssignVariableOp"default_policy/conv_value_3/kernel=default_policy/conv_value_3/kernel/Initializer/random_uniform*
dtype0
?
6default_policy/conv_value_3/kernel/Read/ReadVariableOpReadVariableOp"default_policy/conv_value_3/kernel*'
_output_shapes
: ?*
dtype0
?
2default_policy/conv_value_3/bias/Initializer/zerosConst*3
_class)
'%loc:@default_policy/conv_value_3/bias*
_output_shapes	
:?*
dtype0*
valueB?*    
?
 default_policy/conv_value_3/biasVarHandleOp*3
_class)
'%loc:@default_policy/conv_value_3/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:?*1
shared_name" default_policy/conv_value_3/bias
?
Adefault_policy/conv_value_3/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp default_policy/conv_value_3/bias*
_output_shapes
: 
?
'default_policy/conv_value_3/bias/AssignAssignVariableOp default_policy/conv_value_3/bias2default_policy/conv_value_3/bias/Initializer/zeros*
dtype0
?
4default_policy/conv_value_3/bias/Read/ReadVariableOpReadVariableOp default_policy/conv_value_3/bias*
_output_shapes	
:?*
dtype0
?
1default_policy/conv_value_3/Conv2D/ReadVariableOpReadVariableOp"default_policy/conv_value_3/kernel*'
_output_shapes
: ?*
dtype0
?
"default_policy/conv_value_3/Conv2DConv2D default_policy/conv_value_2/Relu1default_policy/conv_value_3/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingVALID*
strides
*
use_cudnn_on_gpu(
?
2default_policy/conv_value_3/BiasAdd/ReadVariableOpReadVariableOp default_policy/conv_value_3/bias*
_output_shapes	
:?*
dtype0
?
#default_policy/conv_value_3/BiasAddBiasAdd"default_policy/conv_value_3/Conv2D2default_policy/conv_value_3/BiasAdd/ReadVariableOp*
T0*0
_output_shapes
:??????????*
data_formatNHWC
?
 default_policy/conv_value_3/ReluRelu#default_policy/conv_value_3/BiasAdd*
T0*0
_output_shapes
:??????????
?
Edefault_policy/conv_value_out/kernel/Initializer/random_uniform/shapeConst*7
_class-
+)loc:@default_policy/conv_value_out/kernel*
_output_shapes
:*
dtype0*%
valueB"            
?
Cdefault_policy/conv_value_out/kernel/Initializer/random_uniform/minConst*7
_class-
+)loc:@default_policy/conv_value_out/kernel*
_output_shapes
: *
dtype0*
valueB
 *Iv?
?
Cdefault_policy/conv_value_out/kernel/Initializer/random_uniform/maxConst*7
_class-
+)loc:@default_policy/conv_value_out/kernel*
_output_shapes
: *
dtype0*
valueB
 *Iv>
?
Mdefault_policy/conv_value_out/kernel/Initializer/random_uniform/RandomUniformRandomUniformEdefault_policy/conv_value_out/kernel/Initializer/random_uniform/shape*
T0*7
_class-
+)loc:@default_policy/conv_value_out/kernel*'
_output_shapes
:?*
dtype0*

seed *
seed2 
?
Cdefault_policy/conv_value_out/kernel/Initializer/random_uniform/subSubCdefault_policy/conv_value_out/kernel/Initializer/random_uniform/maxCdefault_policy/conv_value_out/kernel/Initializer/random_uniform/min*
T0*7
_class-
+)loc:@default_policy/conv_value_out/kernel*
_output_shapes
: 
?
Cdefault_policy/conv_value_out/kernel/Initializer/random_uniform/mulMulMdefault_policy/conv_value_out/kernel/Initializer/random_uniform/RandomUniformCdefault_policy/conv_value_out/kernel/Initializer/random_uniform/sub*
T0*7
_class-
+)loc:@default_policy/conv_value_out/kernel*'
_output_shapes
:?
?
?default_policy/conv_value_out/kernel/Initializer/random_uniformAddV2Cdefault_policy/conv_value_out/kernel/Initializer/random_uniform/mulCdefault_policy/conv_value_out/kernel/Initializer/random_uniform/min*
T0*7
_class-
+)loc:@default_policy/conv_value_out/kernel*'
_output_shapes
:?
?
$default_policy/conv_value_out/kernelVarHandleOp*7
_class-
+)loc:@default_policy/conv_value_out/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:?*5
shared_name&$default_policy/conv_value_out/kernel
?
Edefault_policy/conv_value_out/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp$default_policy/conv_value_out/kernel*
_output_shapes
: 
?
+default_policy/conv_value_out/kernel/AssignAssignVariableOp$default_policy/conv_value_out/kernel?default_policy/conv_value_out/kernel/Initializer/random_uniform*
dtype0
?
8default_policy/conv_value_out/kernel/Read/ReadVariableOpReadVariableOp$default_policy/conv_value_out/kernel*'
_output_shapes
:?*
dtype0
?
4default_policy/conv_value_out/bias/Initializer/zerosConst*5
_class+
)'loc:@default_policy/conv_value_out/bias*
_output_shapes
:*
dtype0*
valueB*    
?
"default_policy/conv_value_out/biasVarHandleOp*5
_class+
)'loc:@default_policy/conv_value_out/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*3
shared_name$"default_policy/conv_value_out/bias
?
Cdefault_policy/conv_value_out/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp"default_policy/conv_value_out/bias*
_output_shapes
: 
?
)default_policy/conv_value_out/bias/AssignAssignVariableOp"default_policy/conv_value_out/bias4default_policy/conv_value_out/bias/Initializer/zeros*
dtype0
?
6default_policy/conv_value_out/bias/Read/ReadVariableOpReadVariableOp"default_policy/conv_value_out/bias*
_output_shapes
:*
dtype0
?
3default_policy/conv_value_out/Conv2D/ReadVariableOpReadVariableOp$default_policy/conv_value_out/kernel*'
_output_shapes
:?*
dtype0
?
$default_policy/conv_value_out/Conv2DConv2D default_policy/conv_value_3/Relu3default_policy/conv_value_out/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
4default_policy/conv_value_out/BiasAdd/ReadVariableOpReadVariableOp"default_policy/conv_value_out/bias*
_output_shapes
:*
dtype0
?
%default_policy/conv_value_out/BiasAddBiasAdd$default_policy/conv_value_out/Conv2D4default_policy/conv_value_out/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:?????????*
data_formatNHWC
?
default_policy/lambda/SqueezeSqueeze%default_policy/conv_value_out/BiasAdd*
T0*'
_output_shapes
:?????????*
squeeze_dims

p
default_policy/actionPlaceholder*#
_output_shapes
:?????????*
dtype0	*
shape:?????????
?
default_policy/obsPlaceholder*/
_output_shapes
:?????????TT*
dtype0*$
shape:?????????TT
?
default_policy/new_obsPlaceholder*/
_output_shapes
:?????????TT*
dtype0*$
shape:?????????TT
v
default_policy/prev_actionsPlaceholder*#
_output_shapes
:?????????*
dtype0	*
shape:?????????
q
default_policy/rewardsPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
v
default_policy/prev_rewardsPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
o
default_policy/donesPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
p
default_policy/eps_idPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
s
default_policy/unroll_idPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
u
default_policy/agent_indexPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
k
default_policy/tPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
V
default_policy/zerosConst*
_output_shapes
: *
dtype0	*
value	B	 R 
y
default_policy/timestepPlaceholderWithDefaultdefault_policy/zeros*
_output_shapes
: *
dtype0	*
shape: 
c
!default_policy/is_exploring/inputConst*
_output_shapes
: *
dtype0
*
value	B
 Z
?
default_policy/is_exploringPlaceholderWithDefault!default_policy/is_exploring/input*
_output_shapes
: *
dtype0
*
shape: 
b
 default_policy/is_training/inputConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
?
default_policy/is_trainingPlaceholderWithDefault default_policy/is_training/input*
_output_shapes
: *
dtype0
*
shape: 
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
3default_policy/timestep_1/Initializer/initial_valueConst*,
_class"
 loc:@default_policy/timestep_1*
_output_shapes
: *
dtype0	*
value	B	 R 
?
default_policy/timestep_1VarHandleOp*,
_class"
 loc:@default_policy/timestep_1*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0	*
shape: **
shared_namedefault_policy/timestep_1
?
:default_policy/timestep_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpdefault_policy/timestep_1*
_output_shapes
: 
?
 default_policy/timestep_1/AssignAssignVariableOpdefault_policy/timestep_13default_policy/timestep_1/Initializer/initial_value*
dtype0	

-default_policy/timestep_1/Read/ReadVariableOpReadVariableOpdefault_policy/timestep_1*
_output_shapes
: *
dtype0	
m
default_policy/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@n  
?
default_policy/flatten/ReshapeReshapedefault_policy/obsdefault_policy/flatten/Const*
T0*
Tshape0*)
_output_shapes
:???????????
?
default_policy/CastCastdefault_policy/obs*

DstT0*

SrcT0*
Truncate( */
_output_shapes
:?????????TT
?
7default_policy/model/conv_value_1/Conv2D/ReadVariableOpReadVariableOp"default_policy/conv_value_1/kernel*&
_output_shapes
:*
dtype0
?
(default_policy/model/conv_value_1/Conv2DConv2Ddefault_policy/Cast7default_policy/model/conv_value_1/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
8default_policy/model/conv_value_1/BiasAdd/ReadVariableOpReadVariableOp default_policy/conv_value_1/bias*
_output_shapes
:*
dtype0
?
)default_policy/model/conv_value_1/BiasAddBiasAdd(default_policy/model/conv_value_1/Conv2D8default_policy/model/conv_value_1/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:?????????*
data_formatNHWC
?
&default_policy/model/conv_value_1/ReluRelu)default_policy/model/conv_value_1/BiasAdd*
T0*/
_output_shapes
:?????????
?
7default_policy/model/conv_value_2/Conv2D/ReadVariableOpReadVariableOp"default_policy/conv_value_2/kernel*&
_output_shapes
: *
dtype0
?
(default_policy/model/conv_value_2/Conv2DConv2D&default_policy/model/conv_value_1/Relu7default_policy/model/conv_value_2/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? *
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
8default_policy/model/conv_value_2/BiasAdd/ReadVariableOpReadVariableOp default_policy/conv_value_2/bias*
_output_shapes
: *
dtype0
?
)default_policy/model/conv_value_2/BiasAddBiasAdd(default_policy/model/conv_value_2/Conv2D8default_policy/model/conv_value_2/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:????????? *
data_formatNHWC
?
&default_policy/model/conv_value_2/ReluRelu)default_policy/model/conv_value_2/BiasAdd*
T0*/
_output_shapes
:????????? 
?
0default_policy/model/conv1/Conv2D/ReadVariableOpReadVariableOpdefault_policy/conv1/kernel*&
_output_shapes
:*
dtype0
?
!default_policy/model/conv1/Conv2DConv2Ddefault_policy/Cast0default_policy/model/conv1/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
1default_policy/model/conv1/BiasAdd/ReadVariableOpReadVariableOpdefault_policy/conv1/bias*
_output_shapes
:*
dtype0
?
"default_policy/model/conv1/BiasAddBiasAdd!default_policy/model/conv1/Conv2D1default_policy/model/conv1/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:?????????*
data_formatNHWC
?
default_policy/model/conv1/ReluRelu"default_policy/model/conv1/BiasAdd*
T0*/
_output_shapes
:?????????
?
7default_policy/model/conv_value_3/Conv2D/ReadVariableOpReadVariableOp"default_policy/conv_value_3/kernel*'
_output_shapes
: ?*
dtype0
?
(default_policy/model/conv_value_3/Conv2DConv2D&default_policy/model/conv_value_2/Relu7default_policy/model/conv_value_3/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingVALID*
strides
*
use_cudnn_on_gpu(
?
8default_policy/model/conv_value_3/BiasAdd/ReadVariableOpReadVariableOp default_policy/conv_value_3/bias*
_output_shapes	
:?*
dtype0
?
)default_policy/model/conv_value_3/BiasAddBiasAdd(default_policy/model/conv_value_3/Conv2D8default_policy/model/conv_value_3/BiasAdd/ReadVariableOp*
T0*0
_output_shapes
:??????????*
data_formatNHWC
?
&default_policy/model/conv_value_3/ReluRelu)default_policy/model/conv_value_3/BiasAdd*
T0*0
_output_shapes
:??????????
?
0default_policy/model/conv2/Conv2D/ReadVariableOpReadVariableOpdefault_policy/conv2/kernel*&
_output_shapes
: *
dtype0
?
!default_policy/model/conv2/Conv2DConv2Ddefault_policy/model/conv1/Relu0default_policy/model/conv2/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? *
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
1default_policy/model/conv2/BiasAdd/ReadVariableOpReadVariableOpdefault_policy/conv2/bias*
_output_shapes
: *
dtype0
?
"default_policy/model/conv2/BiasAddBiasAdd!default_policy/model/conv2/Conv2D1default_policy/model/conv2/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:????????? *
data_formatNHWC
?
default_policy/model/conv2/ReluRelu"default_policy/model/conv2/BiasAdd*
T0*/
_output_shapes
:????????? 
?
9default_policy/model/conv_value_out/Conv2D/ReadVariableOpReadVariableOp$default_policy/conv_value_out/kernel*'
_output_shapes
:?*
dtype0
?
*default_policy/model/conv_value_out/Conv2DConv2D&default_policy/model/conv_value_3/Relu9default_policy/model/conv_value_out/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
:default_policy/model/conv_value_out/BiasAdd/ReadVariableOpReadVariableOp"default_policy/conv_value_out/bias*
_output_shapes
:*
dtype0
?
+default_policy/model/conv_value_out/BiasAddBiasAdd*default_policy/model/conv_value_out/Conv2D:default_policy/model/conv_value_out/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:?????????*
data_formatNHWC
?
0default_policy/model/conv3/Conv2D/ReadVariableOpReadVariableOpdefault_policy/conv3/kernel*'
_output_shapes
: ?*
dtype0
?
!default_policy/model/conv3/Conv2DConv2Ddefault_policy/model/conv2/Relu0default_policy/model/conv3/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingVALID*
strides
*
use_cudnn_on_gpu(
?
1default_policy/model/conv3/BiasAdd/ReadVariableOpReadVariableOpdefault_policy/conv3/bias*
_output_shapes	
:?*
dtype0
?
"default_policy/model/conv3/BiasAddBiasAdd!default_policy/model/conv3/Conv2D1default_policy/model/conv3/BiasAdd/ReadVariableOp*
T0*0
_output_shapes
:??????????*
data_formatNHWC
?
default_policy/model/conv3/ReluRelu"default_policy/model/conv3/BiasAdd*
T0*0
_output_shapes
:??????????
?
#default_policy/model/lambda/SqueezeSqueeze+default_policy/model/conv_value_out/BiasAdd*
T0*'
_output_shapes
:?????????*
squeeze_dims

?
3default_policy/model/conv_out/Conv2D/ReadVariableOpReadVariableOpdefault_policy/conv_out/kernel*'
_output_shapes
:?*
dtype0
?
$default_policy/model/conv_out/Conv2DConv2Ddefault_policy/model/conv3/Relu3default_policy/model/conv_out/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
4default_policy/model/conv_out/BiasAdd/ReadVariableOpReadVariableOpdefault_policy/conv_out/bias*
_output_shapes
:*
dtype0
?
%default_policy/model/conv_out/BiasAddBiasAdd$default_policy/model/conv_out/Conv2D4default_policy/model/conv_out/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:?????????*
data_formatNHWC
?
default_policy/SqueezeSqueeze%default_policy/model/conv_out/BiasAdd*
T0*'
_output_shapes
:?????????*
squeeze_dims

]
default_policy/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
default_policy/truedivRealDivdefault_policy/Squeezedefault_policy/truediv/y*
T0*'
_output_shapes
:?????????
t
2default_policy/categorical/Multinomial/num_samplesConst*
_output_shapes
: *
dtype0*
value	B :
?
&default_policy/categorical/MultinomialMultinomialdefault_policy/truediv2default_policy/categorical/Multinomial/num_samples*
T0*'
_output_shapes
:?????????*
output_dtype0	*

seed *
seed2 
?
default_policy/Squeeze_1Squeeze&default_policy/categorical/Multinomial*
T0	*#
_output_shapes
:?????????*
squeeze_dims

?
default_policy/Cast_1Castdefault_policy/Squeeze_1*

DstT0*

SrcT0	*
Truncate( *#
_output_shapes
:?????????
?
8default_policy/SparseSoftmaxCrossEntropyWithLogits/ShapeShapedefault_policy/Cast_1*
T0*
_output_shapes
:*
out_type0
?
Vdefault_policy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsdefault_policy/truedivdefault_policy/Cast_1*
T0*
Tlabels0*6
_output_shapes$
":?????????:?????????
?
default_policy/NegNegVdefault_policy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*#
_output_shapes
:?????????
o
default_policy/ReadVariableOpReadVariableOpdefault_policy/timestep_1*
_output_shapes
: *
dtype0	
V
default_policy/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R
q
default_policy/addAddV2default_policy/ReadVariableOpdefault_policy/add/y*
T0	*
_output_shapes
: 
W
default_policy/Less/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
g
default_policy/LessLessdefault_policy/adddefault_policy/Less/y*
T0	*
_output_shapes
: 
q
default_policy/cond/SwitchSwitchdefault_policy/Lessdefault_policy/Less*
T0
*
_output_shapes
: : 
g
default_policy/cond/switch_tIdentitydefault_policy/cond/Switch:1*
T0
*
_output_shapes
: 
e
default_policy/cond/switch_fIdentitydefault_policy/cond/Switch*
T0
*
_output_shapes
: 
]
default_policy/cond/pred_idIdentitydefault_policy/Less*
T0
*
_output_shapes
: 
z
default_policy/cond/ConstConst^default_policy/cond/switch_t*
_output_shapes
: *
dtype0
*
value	B
 Z
?
default_policy/cond/cond/SwitchSwitchdefault_policy/cond/Constdefault_policy/cond/Const*
T0
*
_output_shapes
: : 
q
!default_policy/cond/cond/switch_tIdentity!default_policy/cond/cond/Switch:1*
T0
*
_output_shapes
: 
o
!default_policy/cond/cond/switch_fIdentitydefault_policy/cond/cond/Switch*
T0
*
_output_shapes
: 
h
 default_policy/cond/cond/pred_idIdentitydefault_policy/cond/Const*
T0
*
_output_shapes
: 
?
default_policy/cond/cond/ShapeShape)default_policy/cond/cond/Shape/Switch_1:1*
T0*
_output_shapes
:*
out_type0
?
%default_policy/cond/cond/Shape/SwitchSwitchdefault_policy/truedivdefault_policy/cond/pred_id*
T0*)
_class
loc:@default_policy/truediv*:
_output_shapes(
&:?????????:?????????
?
'default_policy/cond/cond/Shape/Switch_1Switch'default_policy/cond/cond/Shape/Switch:1 default_policy/cond/cond/pred_id*
T0*)
_class
loc:@default_policy/truediv*:
_output_shapes(
&:?????????:?????????
?
,default_policy/cond/cond/strided_slice/stackConst"^default_policy/cond/cond/switch_t*
_output_shapes
:*
dtype0*
valueB: 
?
.default_policy/cond/cond/strided_slice/stack_1Const"^default_policy/cond/cond/switch_t*
_output_shapes
:*
dtype0*
valueB:
?
.default_policy/cond/cond/strided_slice/stack_2Const"^default_policy/cond/cond/switch_t*
_output_shapes
:*
dtype0*
valueB:
?
&default_policy/cond/cond/strided_sliceStridedSlicedefault_policy/cond/cond/Shape,default_policy/cond/cond/strided_slice/stack.default_policy/cond/cond/strided_slice/stack_1.default_policy/cond/cond/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
?
-default_policy/cond/cond/random_uniform/shapePack&default_policy/cond/cond/strided_slice*
N*
T0*
_output_shapes
:*

axis 
?
+default_policy/cond/cond/random_uniform/minConst"^default_policy/cond/cond/switch_t*
_output_shapes
: *
dtype0	*
value	B	 R 
?
+default_policy/cond/cond/random_uniform/maxConst"^default_policy/cond/cond/switch_t*
_output_shapes
: *
dtype0	*
value	B	 R
?
'default_policy/cond/cond/random_uniformRandomUniformInt-default_policy/cond/cond/random_uniform/shape+default_policy/cond/cond/random_uniform/min+default_policy/cond/cond/random_uniform/max*
T0*

Tout0	*#
_output_shapes
:?????????*

seed *
seed2 
?
)default_policy/cond/cond/ArgMax/dimensionConst"^default_policy/cond/cond/switch_f*
_output_shapes
: *
dtype0*
value	B :
?
default_policy/cond/cond/ArgMaxArgMax&default_policy/cond/cond/ArgMax/Switch)default_policy/cond/cond/ArgMax/dimension*
T0*

Tidx0*#
_output_shapes
:?????????*
output_type0	
?
&default_policy/cond/cond/ArgMax/SwitchSwitch'default_policy/cond/cond/Shape/Switch:1 default_policy/cond/cond/pred_id*
T0*)
_class
loc:@default_policy/truediv*:
_output_shapes(
&:?????????:?????????
?
default_policy/cond/cond/MergeMergedefault_policy/cond/cond/ArgMax'default_policy/cond/cond/random_uniform*
N*
T0	*%
_output_shapes
:?????????: 
?
$default_policy/cond/zeros_like/ShapeShapedefault_policy/cond/cond/Merge*
T0	*
_output_shapes
:*
out_type0
?
$default_policy/cond/zeros_like/ConstConst^default_policy/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
?
default_policy/cond/zeros_likeFill$default_policy/cond/zeros_like/Shape$default_policy/cond/zeros_like/Const*
T0*#
_output_shapes
:?????????*

index_type0
?
default_policy/cond/Switch_1Switchdefault_policy/Squeeze_1default_policy/cond/pred_id*
T0	*+
_class!
loc:@default_policy/Squeeze_1*2
_output_shapes 
:?????????:?????????
?
default_policy/cond/MergeMergedefault_policy/cond/Switch_1default_policy/cond/cond/Merge*
N*
T0	*%
_output_shapes
:?????????: 
a
default_policy/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
?
default_policy/ArgMaxArgMaxdefault_policy/truedivdefault_policy/ArgMax/dimension*
T0*

Tidx0*#
_output_shapes
:?????????*
output_type0	
?
default_policy/cond_1/SwitchSwitchdefault_policy/is_exploringdefault_policy/is_exploring*
T0
*
_output_shapes
: : 
k
default_policy/cond_1/switch_tIdentitydefault_policy/cond_1/Switch:1*
T0
*
_output_shapes
: 
i
default_policy/cond_1/switch_fIdentitydefault_policy/cond_1/Switch*
T0
*
_output_shapes
: 
g
default_policy/cond_1/pred_idIdentitydefault_policy/is_exploring*
T0
*
_output_shapes
: 
?
default_policy/cond_1/Switch_1Switchdefault_policy/cond/Mergedefault_policy/cond_1/pred_id*
T0	*,
_class"
 loc:@default_policy/cond/Merge*2
_output_shapes 
:?????????:?????????
?
default_policy/cond_1/Switch_2Switchdefault_policy/ArgMaxdefault_policy/cond_1/pred_id*
T0	*(
_class
loc:@default_policy/ArgMax*2
_output_shapes 
:?????????:?????????
?
default_policy/cond_1/MergeMergedefault_policy/cond_1/Switch_2 default_policy/cond_1/Switch_1:1*
N*
T0	*%
_output_shapes
:?????????: 
_
default_policy/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 

default_policy/GreaterEqualGreaterEqualdefault_policy/adddefault_policy/GreaterEqual/y*
T0	*
_output_shapes
: 
y
default_policy/LogicalAnd
LogicalAnddefault_policy/is_exploringdefault_policy/GreaterEqual*
_output_shapes
: 

default_policy/cond_2/SwitchSwitchdefault_policy/LogicalAnddefault_policy/LogicalAnd*
T0
*
_output_shapes
: : 
k
default_policy/cond_2/switch_tIdentitydefault_policy/cond_2/Switch:1*
T0
*
_output_shapes
: 
i
default_policy/cond_2/switch_fIdentitydefault_policy/cond_2/Switch*
T0
*
_output_shapes
: 
e
default_policy/cond_2/pred_idIdentitydefault_policy/LogicalAnd*
T0
*
_output_shapes
: 
?
default_policy/cond_2/Switch_1Switchdefault_policy/Negdefault_policy/cond_2/pred_id*
T0*%
_class
loc:@default_policy/Neg*2
_output_shapes 
:?????????:?????????
?
&default_policy/cond_2/zeros_like/ShapeShape-default_policy/cond_2/zeros_like/Shape/Switch*
T0	*
_output_shapes
:*
out_type0
?
-default_policy/cond_2/zeros_like/Shape/SwitchSwitchdefault_policy/ArgMaxdefault_policy/cond_2/pred_id*
T0	*(
_class
loc:@default_policy/ArgMax*2
_output_shapes 
:?????????:?????????
?
&default_policy/cond_2/zeros_like/ConstConst^default_policy/cond_2/switch_f*
_output_shapes
: *
dtype0*
valueB
 *    
?
 default_policy/cond_2/zeros_likeFill&default_policy/cond_2/zeros_like/Shape&default_policy/cond_2/zeros_like/Const*
T0*#
_output_shapes
:?????????*

index_type0
?
default_policy/cond_2/MergeMerge default_policy/cond_2/zeros_like default_policy/cond_2/Switch_1:1*
N*
T0*%
_output_shapes
:?????????: 
t
default_policy/AssignVariableOpAssignVariableOpdefault_policy/timestep_1default_policy/timestep*
dtype0	
?
default_policy/ReadVariableOp_1ReadVariableOpdefault_policy/timestep_1 ^default_policy/AssignVariableOp*
_output_shapes
: *
dtype0	
d
default_policy/ExpExpdefault_policy/cond_2/Merge*
T0*#
_output_shapes
:?????????
_
default_policy/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
default_policy/truediv_1RealDivdefault_policy/Squeezedefault_policy/truediv_1/y*
T0*'
_output_shapes
:?????????
v
4default_policy/categorical_1/Multinomial/num_samplesConst*
_output_shapes
: *
dtype0*
value	B :
?
(default_policy/categorical_1/MultinomialMultinomialdefault_policy/truediv_14default_policy/categorical_1/Multinomial/num_samples*
T0*'
_output_shapes
:?????????*
output_dtype0	*

seed *
seed2 
?
default_policy/Squeeze_2Squeeze(default_policy/categorical_1/Multinomial*
T0	*#
_output_shapes
:?????????*
squeeze_dims

?
default_policy/Cast_2Castdefault_policy/Squeeze_2*

DstT0*

SrcT0	*
Truncate( *#
_output_shapes
:?????????
?
:default_policy/SparseSoftmaxCrossEntropyWithLogits_1/ShapeShapedefault_policy/Cast_2*
T0*
_output_shapes
:*
out_type0
?
Xdefault_policy/SparseSoftmaxCrossEntropyWithLogits_1/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsdefault_policy/truediv_1default_policy/Cast_2*
T0*
Tlabels0*6
_output_shapes$
":?????????:?????????
?
default_policy/Neg_1NegXdefault_policy/SparseSoftmaxCrossEntropyWithLogits_1/SparseSoftmaxCrossEntropyWithLogits*
T0*#
_output_shapes
:?????????
?
default_policy/Cast_3Castdefault_policy/action*

DstT0*

SrcT0	*
Truncate( *#
_output_shapes
:?????????
?
:default_policy/SparseSoftmaxCrossEntropyWithLogits_2/ShapeShapedefault_policy/Cast_3*
T0*
_output_shapes
:*
out_type0
?
Xdefault_policy/SparseSoftmaxCrossEntropyWithLogits_2/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsdefault_policy/truediv_1default_policy/Cast_3*
T0*
Tlabels0*6
_output_shapes$
":?????????:?????????
?
default_policy/Neg_2NegXdefault_policy/SparseSoftmaxCrossEntropyWithLogits_2/SparseSoftmaxCrossEntropyWithLogits*
T0*#
_output_shapes
:?????????
L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *??L>
?
1default_policy/kl_coeff/Initializer/initial_valueConst**
_class 
loc:@default_policy/kl_coeff*
_output_shapes
: *
dtype0*
valueB
 *??L>
?
default_policy/kl_coeffVarHandleOp**
_class 
loc:@default_policy/kl_coeff*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *(
shared_namedefault_policy/kl_coeff

8default_policy/kl_coeff/IsInitialized/VarIsInitializedOpVarIsInitializedOpdefault_policy/kl_coeff*
_output_shapes
: 
?
default_policy/kl_coeff/AssignAssignVariableOpdefault_policy/kl_coeff1default_policy/kl_coeff/Initializer/initial_value*
dtype0
{
+default_policy/kl_coeff/Read/ReadVariableOpReadVariableOpdefault_policy/kl_coeff*
_output_shapes
: *
dtype0
^
default_policy/kl_coeff_1Placeholder*
_output_shapes
:*
dtype0*
shape:
v
!default_policy/AssignVariableOp_1AssignVariableOpdefault_policy/kl_coeffdefault_policy/kl_coeff_1*
dtype0
L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *    
?
6default_policy/entropy_coeff/Initializer/initial_valueConst*/
_class%
#!loc:@default_policy/entropy_coeff*
_output_shapes
: *
dtype0*
valueB
 *    
?
default_policy/entropy_coeffVarHandleOp*/
_class%
#!loc:@default_policy/entropy_coeff*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *-
shared_namedefault_policy/entropy_coeff
?
=default_policy/entropy_coeff/IsInitialized/VarIsInitializedOpVarIsInitializedOpdefault_policy/entropy_coeff*
_output_shapes
: 
?
#default_policy/entropy_coeff/AssignAssignVariableOpdefault_policy/entropy_coeff6default_policy/entropy_coeff/Initializer/initial_value*
dtype0
?
0default_policy/entropy_coeff/Read/ReadVariableOpReadVariableOpdefault_policy/entropy_coeff*
_output_shapes
: *
dtype0
?
+default_policy/lr/Initializer/initial_valueConst*$
_class
loc:@default_policy/lr*
_output_shapes
: *
dtype0*
valueB
 *?Q8
?
default_policy/lrVarHandleOp*$
_class
loc:@default_policy/lr*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *"
shared_namedefault_policy/lr
s
2default_policy/lr/IsInitialized/VarIsInitializedOpVarIsInitializedOpdefault_policy/lr*
_output_shapes
: 
y
default_policy/lr/AssignAssignVariableOpdefault_policy/lr+default_policy/lr/Initializer/initial_value*
dtype0
o
%default_policy/lr/Read/ReadVariableOpReadVariableOpdefault_policy/lr*
_output_shapes
: *
dtype0
o
default_policy/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
default_policy/ReshapeReshape#default_policy/model/lambda/Squeezedefault_policy/Reshape/shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
default_policy/initNoOp!^default_policy/conv1/bias/Assign#^default_policy/conv1/kernel/Assign!^default_policy/conv2/bias/Assign#^default_policy/conv2/kernel/Assign!^default_policy/conv3/bias/Assign#^default_policy/conv3/kernel/Assign$^default_policy/conv_out/bias/Assign&^default_policy/conv_out/kernel/Assign(^default_policy/conv_value_1/bias/Assign*^default_policy/conv_value_1/kernel/Assign(^default_policy/conv_value_2/bias/Assign*^default_policy/conv_value_2/kernel/Assign(^default_policy/conv_value_3/bias/Assign*^default_policy/conv_value_3/kernel/Assign*^default_policy/conv_value_out/bias/Assign,^default_policy/conv_value_out/kernel/Assign$^default_policy/entropy_coeff/Assign^default_policy/kl_coeff/Assign^default_policy/lr/Assign!^default_policy/timestep_1/Assign
u
default_policy/action_probPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
u
default_policy/action_logpPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
!default_policy/action_dist_inputsPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
r
default_policy/vf_predsPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
u
default_policy/obs_1Placeholder*&
_output_shapes
:TT*
dtype0*
shape:TT
`
default_policy/seq_lensPlaceholder*
_output_shapes
:*
dtype0*
shape:
o
default_policy/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@n  
?
 default_policy/flatten_1/ReshapeReshapedefault_policy/obs_1default_policy/flatten_1/Const*
T0*
Tshape0* 
_output_shapes
:
??
?
default_policy/Cast_4Castdefault_policy/obs_1*

DstT0*

SrcT0*
Truncate( *&
_output_shapes
:TT
?
9default_policy/model_1/conv_value_1/Conv2D/ReadVariableOpReadVariableOp"default_policy/conv_value_1/kernel*&
_output_shapes
:*
dtype0
?
*default_policy/model_1/conv_value_1/Conv2DConv2Ddefault_policy/Cast_49default_policy/model_1/conv_value_1/Conv2D/ReadVariableOp*
T0*&
_output_shapes
:*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
:default_policy/model_1/conv_value_1/BiasAdd/ReadVariableOpReadVariableOp default_policy/conv_value_1/bias*
_output_shapes
:*
dtype0
?
+default_policy/model_1/conv_value_1/BiasAddBiasAdd*default_policy/model_1/conv_value_1/Conv2D:default_policy/model_1/conv_value_1/BiasAdd/ReadVariableOp*
T0*&
_output_shapes
:*
data_formatNHWC
?
(default_policy/model_1/conv_value_1/ReluRelu+default_policy/model_1/conv_value_1/BiasAdd*
T0*&
_output_shapes
:
?
9default_policy/model_1/conv_value_2/Conv2D/ReadVariableOpReadVariableOp"default_policy/conv_value_2/kernel*&
_output_shapes
: *
dtype0
?
*default_policy/model_1/conv_value_2/Conv2DConv2D(default_policy/model_1/conv_value_1/Relu9default_policy/model_1/conv_value_2/Conv2D/ReadVariableOp*
T0*&
_output_shapes
: *
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
:default_policy/model_1/conv_value_2/BiasAdd/ReadVariableOpReadVariableOp default_policy/conv_value_2/bias*
_output_shapes
: *
dtype0
?
+default_policy/model_1/conv_value_2/BiasAddBiasAdd*default_policy/model_1/conv_value_2/Conv2D:default_policy/model_1/conv_value_2/BiasAdd/ReadVariableOp*
T0*&
_output_shapes
: *
data_formatNHWC
?
(default_policy/model_1/conv_value_2/ReluRelu+default_policy/model_1/conv_value_2/BiasAdd*
T0*&
_output_shapes
: 
?
2default_policy/model_1/conv1/Conv2D/ReadVariableOpReadVariableOpdefault_policy/conv1/kernel*&
_output_shapes
:*
dtype0
?
#default_policy/model_1/conv1/Conv2DConv2Ddefault_policy/Cast_42default_policy/model_1/conv1/Conv2D/ReadVariableOp*
T0*&
_output_shapes
:*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
3default_policy/model_1/conv1/BiasAdd/ReadVariableOpReadVariableOpdefault_policy/conv1/bias*
_output_shapes
:*
dtype0
?
$default_policy/model_1/conv1/BiasAddBiasAdd#default_policy/model_1/conv1/Conv2D3default_policy/model_1/conv1/BiasAdd/ReadVariableOp*
T0*&
_output_shapes
:*
data_formatNHWC
?
!default_policy/model_1/conv1/ReluRelu$default_policy/model_1/conv1/BiasAdd*
T0*&
_output_shapes
:
?
9default_policy/model_1/conv_value_3/Conv2D/ReadVariableOpReadVariableOp"default_policy/conv_value_3/kernel*'
_output_shapes
: ?*
dtype0
?
*default_policy/model_1/conv_value_3/Conv2DConv2D(default_policy/model_1/conv_value_2/Relu9default_policy/model_1/conv_value_3/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:?*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingVALID*
strides
*
use_cudnn_on_gpu(
?
:default_policy/model_1/conv_value_3/BiasAdd/ReadVariableOpReadVariableOp default_policy/conv_value_3/bias*
_output_shapes	
:?*
dtype0
?
+default_policy/model_1/conv_value_3/BiasAddBiasAdd*default_policy/model_1/conv_value_3/Conv2D:default_policy/model_1/conv_value_3/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:?*
data_formatNHWC
?
(default_policy/model_1/conv_value_3/ReluRelu+default_policy/model_1/conv_value_3/BiasAdd*
T0*'
_output_shapes
:?
?
2default_policy/model_1/conv2/Conv2D/ReadVariableOpReadVariableOpdefault_policy/conv2/kernel*&
_output_shapes
: *
dtype0
?
#default_policy/model_1/conv2/Conv2DConv2D!default_policy/model_1/conv1/Relu2default_policy/model_1/conv2/Conv2D/ReadVariableOp*
T0*&
_output_shapes
: *
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
3default_policy/model_1/conv2/BiasAdd/ReadVariableOpReadVariableOpdefault_policy/conv2/bias*
_output_shapes
: *
dtype0
?
$default_policy/model_1/conv2/BiasAddBiasAdd#default_policy/model_1/conv2/Conv2D3default_policy/model_1/conv2/BiasAdd/ReadVariableOp*
T0*&
_output_shapes
: *
data_formatNHWC
?
!default_policy/model_1/conv2/ReluRelu$default_policy/model_1/conv2/BiasAdd*
T0*&
_output_shapes
: 
?
;default_policy/model_1/conv_value_out/Conv2D/ReadVariableOpReadVariableOp$default_policy/conv_value_out/kernel*'
_output_shapes
:?*
dtype0
?
,default_policy/model_1/conv_value_out/Conv2DConv2D(default_policy/model_1/conv_value_3/Relu;default_policy/model_1/conv_value_out/Conv2D/ReadVariableOp*
T0*&
_output_shapes
:*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
<default_policy/model_1/conv_value_out/BiasAdd/ReadVariableOpReadVariableOp"default_policy/conv_value_out/bias*
_output_shapes
:*
dtype0
?
-default_policy/model_1/conv_value_out/BiasAddBiasAdd,default_policy/model_1/conv_value_out/Conv2D<default_policy/model_1/conv_value_out/BiasAdd/ReadVariableOp*
T0*&
_output_shapes
:*
data_formatNHWC
?
2default_policy/model_1/conv3/Conv2D/ReadVariableOpReadVariableOpdefault_policy/conv3/kernel*'
_output_shapes
: ?*
dtype0
?
#default_policy/model_1/conv3/Conv2DConv2D!default_policy/model_1/conv2/Relu2default_policy/model_1/conv3/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:?*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingVALID*
strides
*
use_cudnn_on_gpu(
?
3default_policy/model_1/conv3/BiasAdd/ReadVariableOpReadVariableOpdefault_policy/conv3/bias*
_output_shapes	
:?*
dtype0
?
$default_policy/model_1/conv3/BiasAddBiasAdd#default_policy/model_1/conv3/Conv2D3default_policy/model_1/conv3/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:?*
data_formatNHWC
?
!default_policy/model_1/conv3/ReluRelu$default_policy/model_1/conv3/BiasAdd*
T0*'
_output_shapes
:?
?
%default_policy/model_1/lambda/SqueezeSqueeze-default_policy/model_1/conv_value_out/BiasAdd*
T0*
_output_shapes

:*
squeeze_dims

?
5default_policy/model_1/conv_out/Conv2D/ReadVariableOpReadVariableOpdefault_policy/conv_out/kernel*'
_output_shapes
:?*
dtype0
?
&default_policy/model_1/conv_out/Conv2DConv2D!default_policy/model_1/conv3/Relu5default_policy/model_1/conv_out/Conv2D/ReadVariableOp*
T0*&
_output_shapes
:*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
6default_policy/model_1/conv_out/BiasAdd/ReadVariableOpReadVariableOpdefault_policy/conv_out/bias*
_output_shapes
:*
dtype0
?
'default_policy/model_1/conv_out/BiasAddBiasAdd&default_policy/model_1/conv_out/Conv2D6default_policy/model_1/conv_out/BiasAdd/ReadVariableOp*
T0*&
_output_shapes
:*
data_formatNHWC
?
default_policy/Squeeze_3Squeeze'default_policy/model_1/conv_out/BiasAdd*
T0*
_output_shapes

:*
squeeze_dims

q
default_policy/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
default_policy/Reshape_1Reshape%default_policy/model_1/lambda/Squeezedefault_policy/Reshape_1/shape*
T0*
Tshape0*
_output_shapes
:
l
"default_policy/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
n
$default_policy/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
n
$default_policy/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
default_policy/strided_sliceStridedSlicedefault_policy/Reshape_1"default_policy/strided_slice/stack$default_policy/strided_slice/stack_1$default_policy/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
t
default_policy/advantagesPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
w
default_policy/value_targetsPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
o
default_policy/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@n  
?
 default_policy/flatten_2/ReshapeReshapedefault_policy/obsdefault_policy/flatten_2/Const*
T0*
Tshape0*)
_output_shapes
:???????????
?
default_policy/Cast_5Castdefault_policy/obs*

DstT0*

SrcT0*
Truncate( */
_output_shapes
:?????????TT
?
9default_policy/model_2/conv_value_1/Conv2D/ReadVariableOpReadVariableOp"default_policy/conv_value_1/kernel*&
_output_shapes
:*
dtype0
?
*default_policy/model_2/conv_value_1/Conv2DConv2Ddefault_policy/Cast_59default_policy/model_2/conv_value_1/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
:default_policy/model_2/conv_value_1/BiasAdd/ReadVariableOpReadVariableOp default_policy/conv_value_1/bias*
_output_shapes
:*
dtype0
?
+default_policy/model_2/conv_value_1/BiasAddBiasAdd*default_policy/model_2/conv_value_1/Conv2D:default_policy/model_2/conv_value_1/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:?????????*
data_formatNHWC
?
(default_policy/model_2/conv_value_1/ReluRelu+default_policy/model_2/conv_value_1/BiasAdd*
T0*/
_output_shapes
:?????????
?
9default_policy/model_2/conv_value_2/Conv2D/ReadVariableOpReadVariableOp"default_policy/conv_value_2/kernel*&
_output_shapes
: *
dtype0
?
*default_policy/model_2/conv_value_2/Conv2DConv2D(default_policy/model_2/conv_value_1/Relu9default_policy/model_2/conv_value_2/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? *
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
:default_policy/model_2/conv_value_2/BiasAdd/ReadVariableOpReadVariableOp default_policy/conv_value_2/bias*
_output_shapes
: *
dtype0
?
+default_policy/model_2/conv_value_2/BiasAddBiasAdd*default_policy/model_2/conv_value_2/Conv2D:default_policy/model_2/conv_value_2/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:????????? *
data_formatNHWC
?
(default_policy/model_2/conv_value_2/ReluRelu+default_policy/model_2/conv_value_2/BiasAdd*
T0*/
_output_shapes
:????????? 
?
2default_policy/model_2/conv1/Conv2D/ReadVariableOpReadVariableOpdefault_policy/conv1/kernel*&
_output_shapes
:*
dtype0
?
#default_policy/model_2/conv1/Conv2DConv2Ddefault_policy/Cast_52default_policy/model_2/conv1/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
3default_policy/model_2/conv1/BiasAdd/ReadVariableOpReadVariableOpdefault_policy/conv1/bias*
_output_shapes
:*
dtype0
?
$default_policy/model_2/conv1/BiasAddBiasAdd#default_policy/model_2/conv1/Conv2D3default_policy/model_2/conv1/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:?????????*
data_formatNHWC
?
!default_policy/model_2/conv1/ReluRelu$default_policy/model_2/conv1/BiasAdd*
T0*/
_output_shapes
:?????????
?
9default_policy/model_2/conv_value_3/Conv2D/ReadVariableOpReadVariableOp"default_policy/conv_value_3/kernel*'
_output_shapes
: ?*
dtype0
?
*default_policy/model_2/conv_value_3/Conv2DConv2D(default_policy/model_2/conv_value_2/Relu9default_policy/model_2/conv_value_3/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingVALID*
strides
*
use_cudnn_on_gpu(
?
:default_policy/model_2/conv_value_3/BiasAdd/ReadVariableOpReadVariableOp default_policy/conv_value_3/bias*
_output_shapes	
:?*
dtype0
?
+default_policy/model_2/conv_value_3/BiasAddBiasAdd*default_policy/model_2/conv_value_3/Conv2D:default_policy/model_2/conv_value_3/BiasAdd/ReadVariableOp*
T0*0
_output_shapes
:??????????*
data_formatNHWC
?
(default_policy/model_2/conv_value_3/ReluRelu+default_policy/model_2/conv_value_3/BiasAdd*
T0*0
_output_shapes
:??????????
?
2default_policy/model_2/conv2/Conv2D/ReadVariableOpReadVariableOpdefault_policy/conv2/kernel*&
_output_shapes
: *
dtype0
?
#default_policy/model_2/conv2/Conv2DConv2D!default_policy/model_2/conv1/Relu2default_policy/model_2/conv2/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? *
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
3default_policy/model_2/conv2/BiasAdd/ReadVariableOpReadVariableOpdefault_policy/conv2/bias*
_output_shapes
: *
dtype0
?
$default_policy/model_2/conv2/BiasAddBiasAdd#default_policy/model_2/conv2/Conv2D3default_policy/model_2/conv2/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:????????? *
data_formatNHWC
?
!default_policy/model_2/conv2/ReluRelu$default_policy/model_2/conv2/BiasAdd*
T0*/
_output_shapes
:????????? 
?
;default_policy/model_2/conv_value_out/Conv2D/ReadVariableOpReadVariableOp$default_policy/conv_value_out/kernel*'
_output_shapes
:?*
dtype0
?
,default_policy/model_2/conv_value_out/Conv2DConv2D(default_policy/model_2/conv_value_3/Relu;default_policy/model_2/conv_value_out/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
<default_policy/model_2/conv_value_out/BiasAdd/ReadVariableOpReadVariableOp"default_policy/conv_value_out/bias*
_output_shapes
:*
dtype0
?
-default_policy/model_2/conv_value_out/BiasAddBiasAdd,default_policy/model_2/conv_value_out/Conv2D<default_policy/model_2/conv_value_out/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:?????????*
data_formatNHWC
?
2default_policy/model_2/conv3/Conv2D/ReadVariableOpReadVariableOpdefault_policy/conv3/kernel*'
_output_shapes
: ?*
dtype0
?
#default_policy/model_2/conv3/Conv2DConv2D!default_policy/model_2/conv2/Relu2default_policy/model_2/conv3/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingVALID*
strides
*
use_cudnn_on_gpu(
?
3default_policy/model_2/conv3/BiasAdd/ReadVariableOpReadVariableOpdefault_policy/conv3/bias*
_output_shapes	
:?*
dtype0
?
$default_policy/model_2/conv3/BiasAddBiasAdd#default_policy/model_2/conv3/Conv2D3default_policy/model_2/conv3/BiasAdd/ReadVariableOp*
T0*0
_output_shapes
:??????????*
data_formatNHWC
?
!default_policy/model_2/conv3/ReluRelu$default_policy/model_2/conv3/BiasAdd*
T0*0
_output_shapes
:??????????
?
%default_policy/model_2/lambda/SqueezeSqueeze-default_policy/model_2/conv_value_out/BiasAdd*
T0*'
_output_shapes
:?????????*
squeeze_dims

?
5default_policy/model_2/conv_out/Conv2D/ReadVariableOpReadVariableOpdefault_policy/conv_out/kernel*'
_output_shapes
:?*
dtype0
?
&default_policy/model_2/conv_out/Conv2DConv2D!default_policy/model_2/conv3/Relu5default_policy/model_2/conv_out/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
6default_policy/model_2/conv_out/BiasAdd/ReadVariableOpReadVariableOpdefault_policy/conv_out/bias*
_output_shapes
:*
dtype0
?
'default_policy/model_2/conv_out/BiasAddBiasAdd&default_policy/model_2/conv_out/Conv2D6default_policy/model_2/conv_out/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:?????????*
data_formatNHWC
?
default_policy/Squeeze_4Squeeze'default_policy/model_2/conv_out/BiasAdd*
T0*'
_output_shapes
:?????????*
squeeze_dims

q
default_policy/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
default_policy/Reshape_2Reshape%default_policy/model_2/lambda/Squeezedefault_policy/Reshape_2/shape*
T0*
Tshape0*#
_output_shapes
:?????????
_
default_policy/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
default_policy/truediv_2RealDivdefault_policy/Squeeze_4default_policy/truediv_2/y*
T0*'
_output_shapes
:?????????
v
4default_policy/categorical_2/Multinomial/num_samplesConst*
_output_shapes
: *
dtype0*
value	B :
?
(default_policy/categorical_2/MultinomialMultinomialdefault_policy/truediv_24default_policy/categorical_2/Multinomial/num_samples*
T0*'
_output_shapes
:?????????*
output_dtype0	*

seed *
seed2 
?
default_policy/Squeeze_5Squeeze(default_policy/categorical_2/Multinomial*
T0	*#
_output_shapes
:?????????*
squeeze_dims

?
default_policy/Cast_6Castdefault_policy/Squeeze_5*

DstT0*

SrcT0	*
Truncate( *#
_output_shapes
:?????????
?
:default_policy/SparseSoftmaxCrossEntropyWithLogits_3/ShapeShapedefault_policy/Cast_6*
T0*
_output_shapes
:*
out_type0
?
Xdefault_policy/SparseSoftmaxCrossEntropyWithLogits_3/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsdefault_policy/truediv_2default_policy/Cast_6*
T0*
Tlabels0*6
_output_shapes$
":?????????:?????????
?
default_policy/Neg_3NegXdefault_policy/SparseSoftmaxCrossEntropyWithLogits_3/SparseSoftmaxCrossEntropyWithLogits*
T0*#
_output_shapes
:?????????
_
default_policy/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
default_policy/truediv_3RealDiv!default_policy/action_dist_inputsdefault_policy/truediv_3/y*
T0*'
_output_shapes
:?????????
v
4default_policy/categorical_3/Multinomial/num_samplesConst*
_output_shapes
: *
dtype0*
value	B :
?
(default_policy/categorical_3/MultinomialMultinomialdefault_policy/truediv_34default_policy/categorical_3/Multinomial/num_samples*
T0*'
_output_shapes
:?????????*
output_dtype0	*

seed *
seed2 
?
default_policy/Squeeze_6Squeeze(default_policy/categorical_3/Multinomial*
T0	*#
_output_shapes
:?????????*
squeeze_dims

?
default_policy/Cast_7Castdefault_policy/Squeeze_6*

DstT0*

SrcT0	*
Truncate( *#
_output_shapes
:?????????
?
:default_policy/SparseSoftmaxCrossEntropyWithLogits_4/ShapeShapedefault_policy/Cast_7*
T0*
_output_shapes
:*
out_type0
?
Xdefault_policy/SparseSoftmaxCrossEntropyWithLogits_4/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsdefault_policy/truediv_3default_policy/Cast_7*
T0*
Tlabels0*6
_output_shapes$
":?????????:?????????
?
default_policy/Neg_4NegXdefault_policy/SparseSoftmaxCrossEntropyWithLogits_4/SparseSoftmaxCrossEntropyWithLogits*
T0*#
_output_shapes
:?????????
?
default_policy/Cast_8Castdefault_policy/action*

DstT0*

SrcT0	*
Truncate( *#
_output_shapes
:?????????
?
:default_policy/SparseSoftmaxCrossEntropyWithLogits_5/ShapeShapedefault_policy/Cast_8*
T0*
_output_shapes
:*
out_type0
?
Xdefault_policy/SparseSoftmaxCrossEntropyWithLogits_5/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsdefault_policy/truediv_2default_policy/Cast_8*
T0*
Tlabels0*6
_output_shapes$
":?????????:?????????
?
default_policy/Neg_5NegXdefault_policy/SparseSoftmaxCrossEntropyWithLogits_5/SparseSoftmaxCrossEntropyWithLogits*
T0*#
_output_shapes
:?????????
y
default_policy/subSubdefault_policy/Neg_5default_policy/action_logp*
T0*#
_output_shapes
:?????????
]
default_policy/Exp_1Expdefault_policy/sub*
T0*#
_output_shapes
:?????????
f
$default_policy/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
?
default_policy/MaxMaxdefault_policy/truediv_3$default_policy/Max/reduction_indices*
T0*

Tidx0*'
_output_shapes
:?????????*
	keep_dims(
{
default_policy/sub_1Subdefault_policy/truediv_3default_policy/Max*
T0*'
_output_shapes
:?????????
h
&default_policy/Max_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
?
default_policy/Max_1Maxdefault_policy/truediv_2&default_policy/Max_1/reduction_indices*
T0*

Tidx0*'
_output_shapes
:?????????*
	keep_dims(
}
default_policy/sub_2Subdefault_policy/truediv_2default_policy/Max_1*
T0*'
_output_shapes
:?????????
c
default_policy/Exp_2Expdefault_policy/sub_1*
T0*'
_output_shapes
:?????????
c
default_policy/Exp_3Expdefault_policy/sub_2*
T0*'
_output_shapes
:?????????
f
$default_policy/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
?
default_policy/SumSumdefault_policy/Exp_2$default_policy/Sum/reduction_indices*
T0*

Tidx0*'
_output_shapes
:?????????*
	keep_dims(
h
&default_policy/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
?
default_policy/Sum_1Sumdefault_policy/Exp_3&default_policy/Sum_1/reduction_indices*
T0*

Tidx0*'
_output_shapes
:?????????*
	keep_dims(

default_policy/truediv_4RealDivdefault_policy/Exp_2default_policy/Sum*
T0*'
_output_shapes
:?????????
_
default_policy/LogLogdefault_policy/Sum*
T0*'
_output_shapes
:?????????
w
default_policy/sub_3Subdefault_policy/sub_1default_policy/Log*
T0*'
_output_shapes
:?????????
y
default_policy/sub_4Subdefault_policy/sub_3default_policy/sub_2*
T0*'
_output_shapes
:?????????
c
default_policy/Log_1Logdefault_policy/Sum_1*
T0*'
_output_shapes
:?????????
{
default_policy/add_1AddV2default_policy/sub_4default_policy/Log_1*
T0*'
_output_shapes
:?????????
{
default_policy/mulMuldefault_policy/truediv_4default_policy/add_1*
T0*'
_output_shapes
:?????????
h
&default_policy/Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
?
default_policy/Sum_2Sumdefault_policy/mul&default_policy/Sum_2/reduction_indices*
T0*

Tidx0*#
_output_shapes
:?????????*
	keep_dims( 
^
default_policy/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
default_policy/MeanMeandefault_policy/Sum_2default_policy/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
h
&default_policy/Max_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
?
default_policy/Max_2Maxdefault_policy/truediv_2&default_policy/Max_2/reduction_indices*
T0*

Tidx0*'
_output_shapes
:?????????*
	keep_dims(
}
default_policy/sub_5Subdefault_policy/truediv_2default_policy/Max_2*
T0*'
_output_shapes
:?????????
c
default_policy/Exp_4Expdefault_policy/sub_5*
T0*'
_output_shapes
:?????????
h
&default_policy/Sum_3/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
?
default_policy/Sum_3Sumdefault_policy/Exp_4&default_policy/Sum_3/reduction_indices*
T0*

Tidx0*'
_output_shapes
:?????????*
	keep_dims(
?
default_policy/truediv_5RealDivdefault_policy/Exp_4default_policy/Sum_3*
T0*'
_output_shapes
:?????????
c
default_policy/Log_2Logdefault_policy/Sum_3*
T0*'
_output_shapes
:?????????
y
default_policy/sub_6Subdefault_policy/Log_2default_policy/sub_5*
T0*'
_output_shapes
:?????????
}
default_policy/mul_1Muldefault_policy/truediv_5default_policy/sub_6*
T0*'
_output_shapes
:?????????
h
&default_policy/Sum_4/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
?
default_policy/Sum_4Sumdefault_policy/mul_1&default_policy/Sum_4/reduction_indices*
T0*

Tidx0*#
_output_shapes
:?????????*
	keep_dims( 
`
default_policy/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
?
default_policy/Mean_1Meandefault_policy/Sum_4default_policy/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
z
default_policy/mul_2Muldefault_policy/advantagesdefault_policy/Exp_1*
T0*#
_output_shapes
:?????????
k
&default_policy/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ff??
?
$default_policy/clip_by_value/MinimumMinimumdefault_policy/Exp_1&default_policy/clip_by_value/Minimum/y*
T0*#
_output_shapes
:?????????
c
default_policy/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?
?
default_policy/clip_by_valueMaximum$default_policy/clip_by_value/Minimumdefault_policy/clip_by_value/y*
T0*#
_output_shapes
:?????????
?
default_policy/mul_3Muldefault_policy/advantagesdefault_policy/clip_by_value*
T0*#
_output_shapes
:?????????
{
default_policy/MinimumMinimumdefault_policy/mul_2default_policy/mul_3*
T0*#
_output_shapes
:?????????
a
default_policy/Neg_6Negdefault_policy/Minimum*
T0*#
_output_shapes
:?????????
`
default_policy/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
?
default_policy/Mean_2Meandefault_policy/Neg_6default_policy/Const_2*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
?
default_policy/sub_7Subdefault_policy/Reshape_2default_policy/value_targets*
T0*#
_output_shapes
:?????????
c
default_policy/SquareSquaredefault_policy/sub_7*
T0*#
_output_shapes
:?????????
|
default_policy/sub_8Subdefault_policy/Reshape_2default_policy/vf_preds*
T0*#
_output_shapes
:?????????
m
(default_policy/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
?
&default_policy/clip_by_value_1/MinimumMinimumdefault_policy/sub_8(default_policy/clip_by_value_1/Minimum/y*
T0*#
_output_shapes
:?????????
e
 default_policy/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
?
default_policy/clip_by_value_1Maximum&default_policy/clip_by_value_1/Minimum default_policy/clip_by_value_1/y*
T0*#
_output_shapes
:?????????
?
default_policy/add_2AddV2default_policy/vf_predsdefault_policy/clip_by_value_1*
T0*#
_output_shapes
:?????????
}
default_policy/sub_9Subdefault_policy/add_2default_policy/value_targets*
T0*#
_output_shapes
:?????????
e
default_policy/Square_1Squaredefault_policy/sub_9*
T0*#
_output_shapes
:?????????

default_policy/MaximumMaximumdefault_policy/Squaredefault_policy/Square_1*
T0*#
_output_shapes
:?????????
`
default_policy/Const_3Const*
_output_shapes
:*
dtype0*
valueB: 
?
default_policy/Mean_3Meandefault_policy/Maximumdefault_policy/Const_3*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
a
default_policy/Neg_7Negdefault_policy/Minimum*
T0*#
_output_shapes
:?????????
o
default_policy/ReadVariableOp_2ReadVariableOpdefault_policy/kl_coeff*
_output_shapes
: *
dtype0
?
default_policy/mul_4Muldefault_policy/ReadVariableOp_2default_policy/Sum_2*
T0*#
_output_shapes
:?????????
w
default_policy/add_3AddV2default_policy/Neg_7default_policy/mul_4*
T0*#
_output_shapes
:?????????
[
default_policy/mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
y
default_policy/mul_5Muldefault_policy/mul_5/xdefault_policy/Maximum*
T0*#
_output_shapes
:?????????
w
default_policy/add_4AddV2default_policy/add_3default_policy/mul_5*
T0*#
_output_shapes
:?????????
t
default_policy/ReadVariableOp_3ReadVariableOpdefault_policy/entropy_coeff*
_output_shapes
: *
dtype0
?
default_policy/mul_6Muldefault_policy/ReadVariableOp_3default_policy/Sum_4*
T0*#
_output_shapes
:?????????
v
default_policy/sub_10Subdefault_policy/add_4default_policy/mul_6*
T0*#
_output_shapes
:?????????
`
default_policy/Const_4Const*
_output_shapes
:*
dtype0*
valueB: 
?
default_policy/Mean_4Meandefault_policy/sub_10default_policy/Const_4*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
t
$default_policy/Cast_9/ReadVariableOpReadVariableOpdefault_policy/kl_coeff*
_output_shapes
: *
dtype0
?
default_policy/Cast_9Cast$default_policy/Cast_9/ReadVariableOp*

DstT0*

SrcT0*
Truncate( *
_output_shapes
: 
o
%default_policy/Cast_10/ReadVariableOpReadVariableOpdefault_policy/lr*
_output_shapes
: *
dtype0
?
default_policy/Cast_10Cast%default_policy/Cast_10/ReadVariableOp*

DstT0*

SrcT0*
Truncate( *
_output_shapes
: 
w
-default_policy/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
?
default_policy/moments/meanMeandefault_policy/value_targets-default_policy/moments/mean/reduction_indices*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
u
#default_policy/moments/StopGradientStopGradientdefault_policy/moments/mean*
T0*
_output_shapes
:
?
(default_policy/moments/SquaredDifferenceSquaredDifferencedefault_policy/value_targets#default_policy/moments/StopGradient*
T0*#
_output_shapes
:?????????
{
1default_policy/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
?
default_policy/moments/varianceMean(default_policy/moments/SquaredDifference1default_policy/moments/variance/reduction_indices*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
~
default_policy/moments/SqueezeSqueezedefault_policy/moments/mean*
T0*
_output_shapes
: *
squeeze_dims
 
?
 default_policy/moments/Squeeze_1Squeezedefault_policy/moments/variance*
T0*
_output_shapes
: *
squeeze_dims
 
?
default_policy/sub_11Subdefault_policy/value_targetsdefault_policy/Reshape_2*
T0*#
_output_shapes
:?????????
y
/default_policy/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
?
default_policy/moments_1/meanMeandefault_policy/sub_11/default_policy/moments_1/mean/reduction_indices*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
y
%default_policy/moments_1/StopGradientStopGradientdefault_policy/moments_1/mean*
T0*
_output_shapes
:
?
*default_policy/moments_1/SquaredDifferenceSquaredDifferencedefault_policy/sub_11%default_policy/moments_1/StopGradient*
T0*#
_output_shapes
:?????????
}
3default_policy/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
?
!default_policy/moments_1/varianceMean*default_policy/moments_1/SquaredDifference3default_policy/moments_1/variance/reduction_indices*
T0*

Tidx0*
_output_shapes
:*
	keep_dims(
?
 default_policy/moments_1/SqueezeSqueezedefault_policy/moments_1/mean*
T0*
_output_shapes
: *
squeeze_dims
 
?
"default_policy/moments_1/Squeeze_1Squeeze!default_policy/moments_1/variance*
T0*
_output_shapes
: *
squeeze_dims
 
?
default_policy/truediv_6RealDiv"default_policy/moments_1/Squeeze_1 default_policy/moments/Squeeze_1*
T0*
_output_shapes
: 
\
default_policy/sub_12/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
p
default_policy/sub_12Subdefault_policy/sub_12/xdefault_policy/truediv_6*
T0*
_output_shapes
: 
_
default_policy/Maximum_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
w
default_policy/Maximum_1Maximumdefault_policy/Maximum_1/xdefault_policy/sub_12*
T0*
_output_shapes
: 
z
%default_policy/Cast_11/ReadVariableOpReadVariableOpdefault_policy/entropy_coeff*
_output_shapes
: *
dtype0
?
default_policy/Cast_11Cast%default_policy/Cast_11/ReadVariableOp*

DstT0*

SrcT0*
Truncate( *
_output_shapes
: 
a
default_policy/gradients/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
m
(default_policy/gradients/grad_ys_0/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
"default_policy/gradients/grad_ys_0Filldefault_policy/gradients/Shape(default_policy/gradients/grad_ys_0/Const*
T0*
_output_shapes
: *

index_type0
?
Adefault_policy/gradients/default_policy/Mean_4_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?
;default_policy/gradients/default_policy/Mean_4_grad/ReshapeReshape"default_policy/gradients/grad_ys_0Adefault_policy/gradients/default_policy/Mean_4_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
?
9default_policy/gradients/default_policy/Mean_4_grad/ShapeShapedefault_policy/sub_10*
T0*
_output_shapes
:*
out_type0
?
8default_policy/gradients/default_policy/Mean_4_grad/TileTile;default_policy/gradients/default_policy/Mean_4_grad/Reshape9default_policy/gradients/default_policy/Mean_4_grad/Shape*
T0*

Tmultiples0*#
_output_shapes
:?????????
?
;default_policy/gradients/default_policy/Mean_4_grad/Shape_1Shapedefault_policy/sub_10*
T0*
_output_shapes
:*
out_type0
~
;default_policy/gradients/default_policy/Mean_4_grad/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 
?
9default_policy/gradients/default_policy/Mean_4_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
8default_policy/gradients/default_policy/Mean_4_grad/ProdProd;default_policy/gradients/default_policy/Mean_4_grad/Shape_19default_policy/gradients/default_policy/Mean_4_grad/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
?
;default_policy/gradients/default_policy/Mean_4_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
?
:default_policy/gradients/default_policy/Mean_4_grad/Prod_1Prod;default_policy/gradients/default_policy/Mean_4_grad/Shape_2;default_policy/gradients/default_policy/Mean_4_grad/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 

=default_policy/gradients/default_policy/Mean_4_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :
?
;default_policy/gradients/default_policy/Mean_4_grad/MaximumMaximum:default_policy/gradients/default_policy/Mean_4_grad/Prod_1=default_policy/gradients/default_policy/Mean_4_grad/Maximum/y*
T0*
_output_shapes
: 
?
<default_policy/gradients/default_policy/Mean_4_grad/floordivFloorDiv8default_policy/gradients/default_policy/Mean_4_grad/Prod;default_policy/gradients/default_policy/Mean_4_grad/Maximum*
T0*
_output_shapes
: 
?
8default_policy/gradients/default_policy/Mean_4_grad/CastCast<default_policy/gradients/default_policy/Mean_4_grad/floordiv*

DstT0*

SrcT0*
Truncate( *
_output_shapes
: 
?
;default_policy/gradients/default_policy/Mean_4_grad/truedivRealDiv8default_policy/gradients/default_policy/Mean_4_grad/Tile8default_policy/gradients/default_policy/Mean_4_grad/Cast*
T0*#
_output_shapes
:?????????
?
9default_policy/gradients/default_policy/sub_10_grad/ShapeShapedefault_policy/add_4*
T0*
_output_shapes
:*
out_type0
?
;default_policy/gradients/default_policy/sub_10_grad/Shape_1Shapedefault_policy/mul_6*
T0*
_output_shapes
:*
out_type0
?
Idefault_policy/gradients/default_policy/sub_10_grad/BroadcastGradientArgsBroadcastGradientArgs9default_policy/gradients/default_policy/sub_10_grad/Shape;default_policy/gradients/default_policy/sub_10_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
7default_policy/gradients/default_policy/sub_10_grad/SumSum;default_policy/gradients/default_policy/Mean_4_grad/truedivIdefault_policy/gradients/default_policy/sub_10_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
;default_policy/gradients/default_policy/sub_10_grad/ReshapeReshape7default_policy/gradients/default_policy/sub_10_grad/Sum9default_policy/gradients/default_policy/sub_10_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
7default_policy/gradients/default_policy/sub_10_grad/NegNeg;default_policy/gradients/default_policy/Mean_4_grad/truediv*
T0*#
_output_shapes
:?????????
?
9default_policy/gradients/default_policy/sub_10_grad/Sum_1Sum7default_policy/gradients/default_policy/sub_10_grad/NegKdefault_policy/gradients/default_policy/sub_10_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
=default_policy/gradients/default_policy/sub_10_grad/Reshape_1Reshape9default_policy/gradients/default_policy/sub_10_grad/Sum_1;default_policy/gradients/default_policy/sub_10_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:?????????
?
Ddefault_policy/gradients/default_policy/sub_10_grad/tuple/group_depsNoOp<^default_policy/gradients/default_policy/sub_10_grad/Reshape>^default_policy/gradients/default_policy/sub_10_grad/Reshape_1
?
Ldefault_policy/gradients/default_policy/sub_10_grad/tuple/control_dependencyIdentity;default_policy/gradients/default_policy/sub_10_grad/ReshapeE^default_policy/gradients/default_policy/sub_10_grad/tuple/group_deps*
T0*N
_classD
B@loc:@default_policy/gradients/default_policy/sub_10_grad/Reshape*#
_output_shapes
:?????????
?
Ndefault_policy/gradients/default_policy/sub_10_grad/tuple/control_dependency_1Identity=default_policy/gradients/default_policy/sub_10_grad/Reshape_1E^default_policy/gradients/default_policy/sub_10_grad/tuple/group_deps*
T0*P
_classF
DBloc:@default_policy/gradients/default_policy/sub_10_grad/Reshape_1*#
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/add_4_grad/ShapeShapedefault_policy/add_3*
T0*
_output_shapes
:*
out_type0
?
:default_policy/gradients/default_policy/add_4_grad/Shape_1Shapedefault_policy/mul_5*
T0*
_output_shapes
:*
out_type0
?
Hdefault_policy/gradients/default_policy/add_4_grad/BroadcastGradientArgsBroadcastGradientArgs8default_policy/gradients/default_policy/add_4_grad/Shape:default_policy/gradients/default_policy/add_4_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
6default_policy/gradients/default_policy/add_4_grad/SumSumLdefault_policy/gradients/default_policy/sub_10_grad/tuple/control_dependencyHdefault_policy/gradients/default_policy/add_4_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
:default_policy/gradients/default_policy/add_4_grad/ReshapeReshape6default_policy/gradients/default_policy/add_4_grad/Sum8default_policy/gradients/default_policy/add_4_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/add_4_grad/Sum_1SumLdefault_policy/gradients/default_policy/sub_10_grad/tuple/control_dependencyJdefault_policy/gradients/default_policy/add_4_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
<default_policy/gradients/default_policy/add_4_grad/Reshape_1Reshape8default_policy/gradients/default_policy/add_4_grad/Sum_1:default_policy/gradients/default_policy/add_4_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:?????????
?
Cdefault_policy/gradients/default_policy/add_4_grad/tuple/group_depsNoOp;^default_policy/gradients/default_policy/add_4_grad/Reshape=^default_policy/gradients/default_policy/add_4_grad/Reshape_1
?
Kdefault_policy/gradients/default_policy/add_4_grad/tuple/control_dependencyIdentity:default_policy/gradients/default_policy/add_4_grad/ReshapeD^default_policy/gradients/default_policy/add_4_grad/tuple/group_deps*
T0*M
_classC
A?loc:@default_policy/gradients/default_policy/add_4_grad/Reshape*#
_output_shapes
:?????????
?
Mdefault_policy/gradients/default_policy/add_4_grad/tuple/control_dependency_1Identity<default_policy/gradients/default_policy/add_4_grad/Reshape_1D^default_policy/gradients/default_policy/add_4_grad/tuple/group_deps*
T0*O
_classE
CAloc:@default_policy/gradients/default_policy/add_4_grad/Reshape_1*#
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/mul_6_grad/ShapeShapedefault_policy/ReadVariableOp_3*
T0*
_output_shapes
: *
out_type0
?
:default_policy/gradients/default_policy/mul_6_grad/Shape_1Shapedefault_policy/Sum_4*
T0*
_output_shapes
:*
out_type0
?
Hdefault_policy/gradients/default_policy/mul_6_grad/BroadcastGradientArgsBroadcastGradientArgs8default_policy/gradients/default_policy/mul_6_grad/Shape:default_policy/gradients/default_policy/mul_6_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
6default_policy/gradients/default_policy/mul_6_grad/MulMulNdefault_policy/gradients/default_policy/sub_10_grad/tuple/control_dependency_1default_policy/Sum_4*
T0*#
_output_shapes
:?????????
?
6default_policy/gradients/default_policy/mul_6_grad/SumSum6default_policy/gradients/default_policy/mul_6_grad/MulHdefault_policy/gradients/default_policy/mul_6_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
:default_policy/gradients/default_policy/mul_6_grad/ReshapeReshape6default_policy/gradients/default_policy/mul_6_grad/Sum8default_policy/gradients/default_policy/mul_6_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
?
8default_policy/gradients/default_policy/mul_6_grad/Mul_1Muldefault_policy/ReadVariableOp_3Ndefault_policy/gradients/default_policy/sub_10_grad/tuple/control_dependency_1*
T0*#
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/mul_6_grad/Sum_1Sum8default_policy/gradients/default_policy/mul_6_grad/Mul_1Jdefault_policy/gradients/default_policy/mul_6_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
<default_policy/gradients/default_policy/mul_6_grad/Reshape_1Reshape8default_policy/gradients/default_policy/mul_6_grad/Sum_1:default_policy/gradients/default_policy/mul_6_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:?????????
?
Cdefault_policy/gradients/default_policy/mul_6_grad/tuple/group_depsNoOp;^default_policy/gradients/default_policy/mul_6_grad/Reshape=^default_policy/gradients/default_policy/mul_6_grad/Reshape_1
?
Kdefault_policy/gradients/default_policy/mul_6_grad/tuple/control_dependencyIdentity:default_policy/gradients/default_policy/mul_6_grad/ReshapeD^default_policy/gradients/default_policy/mul_6_grad/tuple/group_deps*
T0*M
_classC
A?loc:@default_policy/gradients/default_policy/mul_6_grad/Reshape*
_output_shapes
: 
?
Mdefault_policy/gradients/default_policy/mul_6_grad/tuple/control_dependency_1Identity<default_policy/gradients/default_policy/mul_6_grad/Reshape_1D^default_policy/gradients/default_policy/mul_6_grad/tuple/group_deps*
T0*O
_classE
CAloc:@default_policy/gradients/default_policy/mul_6_grad/Reshape_1*#
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/add_3_grad/ShapeShapedefault_policy/Neg_7*
T0*
_output_shapes
:*
out_type0
?
:default_policy/gradients/default_policy/add_3_grad/Shape_1Shapedefault_policy/mul_4*
T0*
_output_shapes
:*
out_type0
?
Hdefault_policy/gradients/default_policy/add_3_grad/BroadcastGradientArgsBroadcastGradientArgs8default_policy/gradients/default_policy/add_3_grad/Shape:default_policy/gradients/default_policy/add_3_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
6default_policy/gradients/default_policy/add_3_grad/SumSumKdefault_policy/gradients/default_policy/add_4_grad/tuple/control_dependencyHdefault_policy/gradients/default_policy/add_3_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
:default_policy/gradients/default_policy/add_3_grad/ReshapeReshape6default_policy/gradients/default_policy/add_3_grad/Sum8default_policy/gradients/default_policy/add_3_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/add_3_grad/Sum_1SumKdefault_policy/gradients/default_policy/add_4_grad/tuple/control_dependencyJdefault_policy/gradients/default_policy/add_3_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
<default_policy/gradients/default_policy/add_3_grad/Reshape_1Reshape8default_policy/gradients/default_policy/add_3_grad/Sum_1:default_policy/gradients/default_policy/add_3_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:?????????
?
Cdefault_policy/gradients/default_policy/add_3_grad/tuple/group_depsNoOp;^default_policy/gradients/default_policy/add_3_grad/Reshape=^default_policy/gradients/default_policy/add_3_grad/Reshape_1
?
Kdefault_policy/gradients/default_policy/add_3_grad/tuple/control_dependencyIdentity:default_policy/gradients/default_policy/add_3_grad/ReshapeD^default_policy/gradients/default_policy/add_3_grad/tuple/group_deps*
T0*M
_classC
A?loc:@default_policy/gradients/default_policy/add_3_grad/Reshape*#
_output_shapes
:?????????
?
Mdefault_policy/gradients/default_policy/add_3_grad/tuple/control_dependency_1Identity<default_policy/gradients/default_policy/add_3_grad/Reshape_1D^default_policy/gradients/default_policy/add_3_grad/tuple/group_deps*
T0*O
_classE
CAloc:@default_policy/gradients/default_policy/add_3_grad/Reshape_1*#
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/mul_5_grad/ShapeShapedefault_policy/mul_5/x*
T0*
_output_shapes
: *
out_type0
?
:default_policy/gradients/default_policy/mul_5_grad/Shape_1Shapedefault_policy/Maximum*
T0*
_output_shapes
:*
out_type0
?
Hdefault_policy/gradients/default_policy/mul_5_grad/BroadcastGradientArgsBroadcastGradientArgs8default_policy/gradients/default_policy/mul_5_grad/Shape:default_policy/gradients/default_policy/mul_5_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
6default_policy/gradients/default_policy/mul_5_grad/MulMulMdefault_policy/gradients/default_policy/add_4_grad/tuple/control_dependency_1default_policy/Maximum*
T0*#
_output_shapes
:?????????
?
6default_policy/gradients/default_policy/mul_5_grad/SumSum6default_policy/gradients/default_policy/mul_5_grad/MulHdefault_policy/gradients/default_policy/mul_5_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
:default_policy/gradients/default_policy/mul_5_grad/ReshapeReshape6default_policy/gradients/default_policy/mul_5_grad/Sum8default_policy/gradients/default_policy/mul_5_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
?
8default_policy/gradients/default_policy/mul_5_grad/Mul_1Muldefault_policy/mul_5/xMdefault_policy/gradients/default_policy/add_4_grad/tuple/control_dependency_1*
T0*#
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/mul_5_grad/Sum_1Sum8default_policy/gradients/default_policy/mul_5_grad/Mul_1Jdefault_policy/gradients/default_policy/mul_5_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
<default_policy/gradients/default_policy/mul_5_grad/Reshape_1Reshape8default_policy/gradients/default_policy/mul_5_grad/Sum_1:default_policy/gradients/default_policy/mul_5_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:?????????
?
Cdefault_policy/gradients/default_policy/mul_5_grad/tuple/group_depsNoOp;^default_policy/gradients/default_policy/mul_5_grad/Reshape=^default_policy/gradients/default_policy/mul_5_grad/Reshape_1
?
Kdefault_policy/gradients/default_policy/mul_5_grad/tuple/control_dependencyIdentity:default_policy/gradients/default_policy/mul_5_grad/ReshapeD^default_policy/gradients/default_policy/mul_5_grad/tuple/group_deps*
T0*M
_classC
A?loc:@default_policy/gradients/default_policy/mul_5_grad/Reshape*
_output_shapes
: 
?
Mdefault_policy/gradients/default_policy/mul_5_grad/tuple/control_dependency_1Identity<default_policy/gradients/default_policy/mul_5_grad/Reshape_1D^default_policy/gradients/default_policy/mul_5_grad/tuple/group_deps*
T0*O
_classE
CAloc:@default_policy/gradients/default_policy/mul_5_grad/Reshape_1*#
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/Sum_4_grad/ShapeShapedefault_policy/mul_1*
T0*
_output_shapes
:*
out_type0
?
7default_policy/gradients/default_policy/Sum_4_grad/SizeConst*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_4_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
?
6default_policy/gradients/default_policy/Sum_4_grad/addAddV2&default_policy/Sum_4/reduction_indices7default_policy/gradients/default_policy/Sum_4_grad/Size*
T0*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_4_grad/Shape*
_output_shapes
: 
?
6default_policy/gradients/default_policy/Sum_4_grad/modFloorMod6default_policy/gradients/default_policy/Sum_4_grad/add7default_policy/gradients/default_policy/Sum_4_grad/Size*
T0*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_4_grad/Shape*
_output_shapes
: 
?
:default_policy/gradients/default_policy/Sum_4_grad/Shape_1Const*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_4_grad/Shape*
_output_shapes
: *
dtype0*
valueB 
?
>default_policy/gradients/default_policy/Sum_4_grad/range/startConst*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_4_grad/Shape*
_output_shapes
: *
dtype0*
value	B : 
?
>default_policy/gradients/default_policy/Sum_4_grad/range/deltaConst*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_4_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
?
8default_policy/gradients/default_policy/Sum_4_grad/rangeRange>default_policy/gradients/default_policy/Sum_4_grad/range/start7default_policy/gradients/default_policy/Sum_4_grad/Size>default_policy/gradients/default_policy/Sum_4_grad/range/delta*

Tidx0*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_4_grad/Shape*
_output_shapes
:
?
=default_policy/gradients/default_policy/Sum_4_grad/ones/ConstConst*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_4_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
?
7default_policy/gradients/default_policy/Sum_4_grad/onesFill:default_policy/gradients/default_policy/Sum_4_grad/Shape_1=default_policy/gradients/default_policy/Sum_4_grad/ones/Const*
T0*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_4_grad/Shape*
_output_shapes
: *

index_type0
?
@default_policy/gradients/default_policy/Sum_4_grad/DynamicStitchDynamicStitch8default_policy/gradients/default_policy/Sum_4_grad/range6default_policy/gradients/default_policy/Sum_4_grad/mod8default_policy/gradients/default_policy/Sum_4_grad/Shape7default_policy/gradients/default_policy/Sum_4_grad/ones*
N*
T0*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_4_grad/Shape*
_output_shapes
:
?
:default_policy/gradients/default_policy/Sum_4_grad/ReshapeReshapeMdefault_policy/gradients/default_policy/mul_6_grad/tuple/control_dependency_1@default_policy/gradients/default_policy/Sum_4_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:??????????????????
?
>default_policy/gradients/default_policy/Sum_4_grad/BroadcastToBroadcastTo:default_policy/gradients/default_policy/Sum_4_grad/Reshape8default_policy/gradients/default_policy/Sum_4_grad/Shape*
T0*

Tidx0*'
_output_shapes
:?????????
?
6default_policy/gradients/default_policy/Neg_7_grad/NegNegKdefault_policy/gradients/default_policy/add_3_grad/tuple/control_dependency*
T0*#
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/mul_4_grad/ShapeShapedefault_policy/ReadVariableOp_2*
T0*
_output_shapes
: *
out_type0
?
:default_policy/gradients/default_policy/mul_4_grad/Shape_1Shapedefault_policy/Sum_2*
T0*
_output_shapes
:*
out_type0
?
Hdefault_policy/gradients/default_policy/mul_4_grad/BroadcastGradientArgsBroadcastGradientArgs8default_policy/gradients/default_policy/mul_4_grad/Shape:default_policy/gradients/default_policy/mul_4_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
6default_policy/gradients/default_policy/mul_4_grad/MulMulMdefault_policy/gradients/default_policy/add_3_grad/tuple/control_dependency_1default_policy/Sum_2*
T0*#
_output_shapes
:?????????
?
6default_policy/gradients/default_policy/mul_4_grad/SumSum6default_policy/gradients/default_policy/mul_4_grad/MulHdefault_policy/gradients/default_policy/mul_4_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
:default_policy/gradients/default_policy/mul_4_grad/ReshapeReshape6default_policy/gradients/default_policy/mul_4_grad/Sum8default_policy/gradients/default_policy/mul_4_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
?
8default_policy/gradients/default_policy/mul_4_grad/Mul_1Muldefault_policy/ReadVariableOp_2Mdefault_policy/gradients/default_policy/add_3_grad/tuple/control_dependency_1*
T0*#
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/mul_4_grad/Sum_1Sum8default_policy/gradients/default_policy/mul_4_grad/Mul_1Jdefault_policy/gradients/default_policy/mul_4_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
<default_policy/gradients/default_policy/mul_4_grad/Reshape_1Reshape8default_policy/gradients/default_policy/mul_4_grad/Sum_1:default_policy/gradients/default_policy/mul_4_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:?????????
?
Cdefault_policy/gradients/default_policy/mul_4_grad/tuple/group_depsNoOp;^default_policy/gradients/default_policy/mul_4_grad/Reshape=^default_policy/gradients/default_policy/mul_4_grad/Reshape_1
?
Kdefault_policy/gradients/default_policy/mul_4_grad/tuple/control_dependencyIdentity:default_policy/gradients/default_policy/mul_4_grad/ReshapeD^default_policy/gradients/default_policy/mul_4_grad/tuple/group_deps*
T0*M
_classC
A?loc:@default_policy/gradients/default_policy/mul_4_grad/Reshape*
_output_shapes
: 
?
Mdefault_policy/gradients/default_policy/mul_4_grad/tuple/control_dependency_1Identity<default_policy/gradients/default_policy/mul_4_grad/Reshape_1D^default_policy/gradients/default_policy/mul_4_grad/tuple/group_deps*
T0*O
_classE
CAloc:@default_policy/gradients/default_policy/mul_4_grad/Reshape_1*#
_output_shapes
:?????????
?
:default_policy/gradients/default_policy/Maximum_grad/ShapeShapedefault_policy/Square*
T0*
_output_shapes
:*
out_type0
?
<default_policy/gradients/default_policy/Maximum_grad/Shape_1Shapedefault_policy/Square_1*
T0*
_output_shapes
:*
out_type0
?
?default_policy/gradients/default_policy/Maximum_grad/zeros_like	ZerosLikeMdefault_policy/gradients/default_policy/mul_5_grad/tuple/control_dependency_1*
T0*#
_output_shapes
:?????????
?
Adefault_policy/gradients/default_policy/Maximum_grad/GreaterEqualGreaterEqualdefault_policy/Squaredefault_policy/Square_1*
T0*#
_output_shapes
:?????????
?
Jdefault_policy/gradients/default_policy/Maximum_grad/BroadcastGradientArgsBroadcastGradientArgs:default_policy/gradients/default_policy/Maximum_grad/Shape<default_policy/gradients/default_policy/Maximum_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
=default_policy/gradients/default_policy/Maximum_grad/SelectV2SelectV2Adefault_policy/gradients/default_policy/Maximum_grad/GreaterEqualMdefault_policy/gradients/default_policy/mul_5_grad/tuple/control_dependency_1?default_policy/gradients/default_policy/Maximum_grad/zeros_like*
T0*#
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/Maximum_grad/SumSum=default_policy/gradients/default_policy/Maximum_grad/SelectV2Jdefault_policy/gradients/default_policy/Maximum_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
<default_policy/gradients/default_policy/Maximum_grad/ReshapeReshape8default_policy/gradients/default_policy/Maximum_grad/Sum:default_policy/gradients/default_policy/Maximum_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
?default_policy/gradients/default_policy/Maximum_grad/SelectV2_1SelectV2Adefault_policy/gradients/default_policy/Maximum_grad/GreaterEqual?default_policy/gradients/default_policy/Maximum_grad/zeros_likeMdefault_policy/gradients/default_policy/mul_5_grad/tuple/control_dependency_1*
T0*#
_output_shapes
:?????????
?
:default_policy/gradients/default_policy/Maximum_grad/Sum_1Sum?default_policy/gradients/default_policy/Maximum_grad/SelectV2_1Ldefault_policy/gradients/default_policy/Maximum_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
>default_policy/gradients/default_policy/Maximum_grad/Reshape_1Reshape:default_policy/gradients/default_policy/Maximum_grad/Sum_1<default_policy/gradients/default_policy/Maximum_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:?????????
?
Edefault_policy/gradients/default_policy/Maximum_grad/tuple/group_depsNoOp=^default_policy/gradients/default_policy/Maximum_grad/Reshape?^default_policy/gradients/default_policy/Maximum_grad/Reshape_1
?
Mdefault_policy/gradients/default_policy/Maximum_grad/tuple/control_dependencyIdentity<default_policy/gradients/default_policy/Maximum_grad/ReshapeF^default_policy/gradients/default_policy/Maximum_grad/tuple/group_deps*
T0*O
_classE
CAloc:@default_policy/gradients/default_policy/Maximum_grad/Reshape*#
_output_shapes
:?????????
?
Odefault_policy/gradients/default_policy/Maximum_grad/tuple/control_dependency_1Identity>default_policy/gradients/default_policy/Maximum_grad/Reshape_1F^default_policy/gradients/default_policy/Maximum_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@default_policy/gradients/default_policy/Maximum_grad/Reshape_1*#
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/mul_1_grad/ShapeShapedefault_policy/truediv_5*
T0*
_output_shapes
:*
out_type0
?
:default_policy/gradients/default_policy/mul_1_grad/Shape_1Shapedefault_policy/sub_6*
T0*
_output_shapes
:*
out_type0
?
Hdefault_policy/gradients/default_policy/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs8default_policy/gradients/default_policy/mul_1_grad/Shape:default_policy/gradients/default_policy/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
6default_policy/gradients/default_policy/mul_1_grad/MulMul>default_policy/gradients/default_policy/Sum_4_grad/BroadcastTodefault_policy/sub_6*
T0*'
_output_shapes
:?????????
?
6default_policy/gradients/default_policy/mul_1_grad/SumSum6default_policy/gradients/default_policy/mul_1_grad/MulHdefault_policy/gradients/default_policy/mul_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
:default_policy/gradients/default_policy/mul_1_grad/ReshapeReshape6default_policy/gradients/default_policy/mul_1_grad/Sum8default_policy/gradients/default_policy/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/mul_1_grad/Mul_1Muldefault_policy/truediv_5>default_policy/gradients/default_policy/Sum_4_grad/BroadcastTo*
T0*'
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/mul_1_grad/Sum_1Sum8default_policy/gradients/default_policy/mul_1_grad/Mul_1Jdefault_policy/gradients/default_policy/mul_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
<default_policy/gradients/default_policy/mul_1_grad/Reshape_1Reshape8default_policy/gradients/default_policy/mul_1_grad/Sum_1:default_policy/gradients/default_policy/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
?
Cdefault_policy/gradients/default_policy/mul_1_grad/tuple/group_depsNoOp;^default_policy/gradients/default_policy/mul_1_grad/Reshape=^default_policy/gradients/default_policy/mul_1_grad/Reshape_1
?
Kdefault_policy/gradients/default_policy/mul_1_grad/tuple/control_dependencyIdentity:default_policy/gradients/default_policy/mul_1_grad/ReshapeD^default_policy/gradients/default_policy/mul_1_grad/tuple/group_deps*
T0*M
_classC
A?loc:@default_policy/gradients/default_policy/mul_1_grad/Reshape*'
_output_shapes
:?????????
?
Mdefault_policy/gradients/default_policy/mul_1_grad/tuple/control_dependency_1Identity<default_policy/gradients/default_policy/mul_1_grad/Reshape_1D^default_policy/gradients/default_policy/mul_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@default_policy/gradients/default_policy/mul_1_grad/Reshape_1*'
_output_shapes
:?????????
?
:default_policy/gradients/default_policy/Minimum_grad/ShapeShapedefault_policy/mul_2*
T0*
_output_shapes
:*
out_type0
?
<default_policy/gradients/default_policy/Minimum_grad/Shape_1Shapedefault_policy/mul_3*
T0*
_output_shapes
:*
out_type0
?
?default_policy/gradients/default_policy/Minimum_grad/zeros_like	ZerosLike6default_policy/gradients/default_policy/Neg_7_grad/Neg*
T0*#
_output_shapes
:?????????
?
>default_policy/gradients/default_policy/Minimum_grad/LessEqual	LessEqualdefault_policy/mul_2default_policy/mul_3*
T0*#
_output_shapes
:?????????
?
Jdefault_policy/gradients/default_policy/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs:default_policy/gradients/default_policy/Minimum_grad/Shape<default_policy/gradients/default_policy/Minimum_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
=default_policy/gradients/default_policy/Minimum_grad/SelectV2SelectV2>default_policy/gradients/default_policy/Minimum_grad/LessEqual6default_policy/gradients/default_policy/Neg_7_grad/Neg?default_policy/gradients/default_policy/Minimum_grad/zeros_like*
T0*#
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/Minimum_grad/SumSum=default_policy/gradients/default_policy/Minimum_grad/SelectV2Jdefault_policy/gradients/default_policy/Minimum_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
<default_policy/gradients/default_policy/Minimum_grad/ReshapeReshape8default_policy/gradients/default_policy/Minimum_grad/Sum:default_policy/gradients/default_policy/Minimum_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
?default_policy/gradients/default_policy/Minimum_grad/SelectV2_1SelectV2>default_policy/gradients/default_policy/Minimum_grad/LessEqual?default_policy/gradients/default_policy/Minimum_grad/zeros_like6default_policy/gradients/default_policy/Neg_7_grad/Neg*
T0*#
_output_shapes
:?????????
?
:default_policy/gradients/default_policy/Minimum_grad/Sum_1Sum?default_policy/gradients/default_policy/Minimum_grad/SelectV2_1Ldefault_policy/gradients/default_policy/Minimum_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
>default_policy/gradients/default_policy/Minimum_grad/Reshape_1Reshape:default_policy/gradients/default_policy/Minimum_grad/Sum_1<default_policy/gradients/default_policy/Minimum_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:?????????
?
Edefault_policy/gradients/default_policy/Minimum_grad/tuple/group_depsNoOp=^default_policy/gradients/default_policy/Minimum_grad/Reshape?^default_policy/gradients/default_policy/Minimum_grad/Reshape_1
?
Mdefault_policy/gradients/default_policy/Minimum_grad/tuple/control_dependencyIdentity<default_policy/gradients/default_policy/Minimum_grad/ReshapeF^default_policy/gradients/default_policy/Minimum_grad/tuple/group_deps*
T0*O
_classE
CAloc:@default_policy/gradients/default_policy/Minimum_grad/Reshape*#
_output_shapes
:?????????
?
Odefault_policy/gradients/default_policy/Minimum_grad/tuple/control_dependency_1Identity>default_policy/gradients/default_policy/Minimum_grad/Reshape_1F^default_policy/gradients/default_policy/Minimum_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@default_policy/gradients/default_policy/Minimum_grad/Reshape_1*#
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/Sum_2_grad/ShapeShapedefault_policy/mul*
T0*
_output_shapes
:*
out_type0
?
7default_policy/gradients/default_policy/Sum_2_grad/SizeConst*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_2_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
?
6default_policy/gradients/default_policy/Sum_2_grad/addAddV2&default_policy/Sum_2/reduction_indices7default_policy/gradients/default_policy/Sum_2_grad/Size*
T0*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_2_grad/Shape*
_output_shapes
: 
?
6default_policy/gradients/default_policy/Sum_2_grad/modFloorMod6default_policy/gradients/default_policy/Sum_2_grad/add7default_policy/gradients/default_policy/Sum_2_grad/Size*
T0*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_2_grad/Shape*
_output_shapes
: 
?
:default_policy/gradients/default_policy/Sum_2_grad/Shape_1Const*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_2_grad/Shape*
_output_shapes
: *
dtype0*
valueB 
?
>default_policy/gradients/default_policy/Sum_2_grad/range/startConst*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_2_grad/Shape*
_output_shapes
: *
dtype0*
value	B : 
?
>default_policy/gradients/default_policy/Sum_2_grad/range/deltaConst*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_2_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
?
8default_policy/gradients/default_policy/Sum_2_grad/rangeRange>default_policy/gradients/default_policy/Sum_2_grad/range/start7default_policy/gradients/default_policy/Sum_2_grad/Size>default_policy/gradients/default_policy/Sum_2_grad/range/delta*

Tidx0*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_2_grad/Shape*
_output_shapes
:
?
=default_policy/gradients/default_policy/Sum_2_grad/ones/ConstConst*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_2_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
?
7default_policy/gradients/default_policy/Sum_2_grad/onesFill:default_policy/gradients/default_policy/Sum_2_grad/Shape_1=default_policy/gradients/default_policy/Sum_2_grad/ones/Const*
T0*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_2_grad/Shape*
_output_shapes
: *

index_type0
?
@default_policy/gradients/default_policy/Sum_2_grad/DynamicStitchDynamicStitch8default_policy/gradients/default_policy/Sum_2_grad/range6default_policy/gradients/default_policy/Sum_2_grad/mod8default_policy/gradients/default_policy/Sum_2_grad/Shape7default_policy/gradients/default_policy/Sum_2_grad/ones*
N*
T0*K
_classA
?=loc:@default_policy/gradients/default_policy/Sum_2_grad/Shape*
_output_shapes
:
?
:default_policy/gradients/default_policy/Sum_2_grad/ReshapeReshapeMdefault_policy/gradients/default_policy/mul_4_grad/tuple/control_dependency_1@default_policy/gradients/default_policy/Sum_2_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:??????????????????
?
>default_policy/gradients/default_policy/Sum_2_grad/BroadcastToBroadcastTo:default_policy/gradients/default_policy/Sum_2_grad/Reshape8default_policy/gradients/default_policy/Sum_2_grad/Shape*
T0*

Tidx0*'
_output_shapes
:?????????
?
9default_policy/gradients/default_policy/Square_grad/ConstConstN^default_policy/gradients/default_policy/Maximum_grad/tuple/control_dependency*
_output_shapes
: *
dtype0*
valueB
 *   @
?
7default_policy/gradients/default_policy/Square_grad/MulMuldefault_policy/sub_79default_policy/gradients/default_policy/Square_grad/Const*
T0*#
_output_shapes
:?????????
?
9default_policy/gradients/default_policy/Square_grad/Mul_1MulMdefault_policy/gradients/default_policy/Maximum_grad/tuple/control_dependency7default_policy/gradients/default_policy/Square_grad/Mul*
T0*#
_output_shapes
:?????????
?
;default_policy/gradients/default_policy/Square_1_grad/ConstConstP^default_policy/gradients/default_policy/Maximum_grad/tuple/control_dependency_1*
_output_shapes
: *
dtype0*
valueB
 *   @
?
9default_policy/gradients/default_policy/Square_1_grad/MulMuldefault_policy/sub_9;default_policy/gradients/default_policy/Square_1_grad/Const*
T0*#
_output_shapes
:?????????
?
;default_policy/gradients/default_policy/Square_1_grad/Mul_1MulOdefault_policy/gradients/default_policy/Maximum_grad/tuple/control_dependency_19default_policy/gradients/default_policy/Square_1_grad/Mul*
T0*#
_output_shapes
:?????????
?
<default_policy/gradients/default_policy/truediv_5_grad/ShapeShapedefault_policy/Exp_4*
T0*
_output_shapes
:*
out_type0
?
>default_policy/gradients/default_policy/truediv_5_grad/Shape_1Shapedefault_policy/Sum_3*
T0*
_output_shapes
:*
out_type0
?
Ldefault_policy/gradients/default_policy/truediv_5_grad/BroadcastGradientArgsBroadcastGradientArgs<default_policy/gradients/default_policy/truediv_5_grad/Shape>default_policy/gradients/default_policy/truediv_5_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
>default_policy/gradients/default_policy/truediv_5_grad/RealDivRealDivKdefault_policy/gradients/default_policy/mul_1_grad/tuple/control_dependencydefault_policy/Sum_3*
T0*'
_output_shapes
:?????????
?
:default_policy/gradients/default_policy/truediv_5_grad/SumSum>default_policy/gradients/default_policy/truediv_5_grad/RealDivLdefault_policy/gradients/default_policy/truediv_5_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
>default_policy/gradients/default_policy/truediv_5_grad/ReshapeReshape:default_policy/gradients/default_policy/truediv_5_grad/Sum<default_policy/gradients/default_policy/truediv_5_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
:default_policy/gradients/default_policy/truediv_5_grad/NegNegdefault_policy/Exp_4*
T0*'
_output_shapes
:?????????
?
@default_policy/gradients/default_policy/truediv_5_grad/RealDiv_1RealDiv:default_policy/gradients/default_policy/truediv_5_grad/Negdefault_policy/Sum_3*
T0*'
_output_shapes
:?????????
?
@default_policy/gradients/default_policy/truediv_5_grad/RealDiv_2RealDiv@default_policy/gradients/default_policy/truediv_5_grad/RealDiv_1default_policy/Sum_3*
T0*'
_output_shapes
:?????????
?
:default_policy/gradients/default_policy/truediv_5_grad/mulMulKdefault_policy/gradients/default_policy/mul_1_grad/tuple/control_dependency@default_policy/gradients/default_policy/truediv_5_grad/RealDiv_2*
T0*'
_output_shapes
:?????????
?
<default_policy/gradients/default_policy/truediv_5_grad/Sum_1Sum:default_policy/gradients/default_policy/truediv_5_grad/mulNdefault_policy/gradients/default_policy/truediv_5_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
@default_policy/gradients/default_policy/truediv_5_grad/Reshape_1Reshape<default_policy/gradients/default_policy/truediv_5_grad/Sum_1>default_policy/gradients/default_policy/truediv_5_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
?
Gdefault_policy/gradients/default_policy/truediv_5_grad/tuple/group_depsNoOp?^default_policy/gradients/default_policy/truediv_5_grad/ReshapeA^default_policy/gradients/default_policy/truediv_5_grad/Reshape_1
?
Odefault_policy/gradients/default_policy/truediv_5_grad/tuple/control_dependencyIdentity>default_policy/gradients/default_policy/truediv_5_grad/ReshapeH^default_policy/gradients/default_policy/truediv_5_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@default_policy/gradients/default_policy/truediv_5_grad/Reshape*'
_output_shapes
:?????????
?
Qdefault_policy/gradients/default_policy/truediv_5_grad/tuple/control_dependency_1Identity@default_policy/gradients/default_policy/truediv_5_grad/Reshape_1H^default_policy/gradients/default_policy/truediv_5_grad/tuple/group_deps*
T0*S
_classI
GEloc:@default_policy/gradients/default_policy/truediv_5_grad/Reshape_1*'
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/sub_6_grad/ShapeShapedefault_policy/Log_2*
T0*
_output_shapes
:*
out_type0
?
:default_policy/gradients/default_policy/sub_6_grad/Shape_1Shapedefault_policy/sub_5*
T0*
_output_shapes
:*
out_type0
?
Hdefault_policy/gradients/default_policy/sub_6_grad/BroadcastGradientArgsBroadcastGradientArgs8default_policy/gradients/default_policy/sub_6_grad/Shape:default_policy/gradients/default_policy/sub_6_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
6default_policy/gradients/default_policy/sub_6_grad/SumSumMdefault_policy/gradients/default_policy/mul_1_grad/tuple/control_dependency_1Hdefault_policy/gradients/default_policy/sub_6_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
:default_policy/gradients/default_policy/sub_6_grad/ReshapeReshape6default_policy/gradients/default_policy/sub_6_grad/Sum8default_policy/gradients/default_policy/sub_6_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
6default_policy/gradients/default_policy/sub_6_grad/NegNegMdefault_policy/gradients/default_policy/mul_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/sub_6_grad/Sum_1Sum6default_policy/gradients/default_policy/sub_6_grad/NegJdefault_policy/gradients/default_policy/sub_6_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
<default_policy/gradients/default_policy/sub_6_grad/Reshape_1Reshape8default_policy/gradients/default_policy/sub_6_grad/Sum_1:default_policy/gradients/default_policy/sub_6_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
?
Cdefault_policy/gradients/default_policy/sub_6_grad/tuple/group_depsNoOp;^default_policy/gradients/default_policy/sub_6_grad/Reshape=^default_policy/gradients/default_policy/sub_6_grad/Reshape_1
?
Kdefault_policy/gradients/default_policy/sub_6_grad/tuple/control_dependencyIdentity:default_policy/gradients/default_policy/sub_6_grad/ReshapeD^default_policy/gradients/default_policy/sub_6_grad/tuple/group_deps*
T0*M
_classC
A?loc:@default_policy/gradients/default_policy/sub_6_grad/Reshape*'
_output_shapes
:?????????
?
Mdefault_policy/gradients/default_policy/sub_6_grad/tuple/control_dependency_1Identity<default_policy/gradients/default_policy/sub_6_grad/Reshape_1D^default_policy/gradients/default_policy/sub_6_grad/tuple/group_deps*
T0*O
_classE
CAloc:@default_policy/gradients/default_policy/sub_6_grad/Reshape_1*'
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/mul_2_grad/ShapeShapedefault_policy/advantages*
T0*
_output_shapes
:*
out_type0
?
:default_policy/gradients/default_policy/mul_2_grad/Shape_1Shapedefault_policy/Exp_1*
T0*
_output_shapes
:*
out_type0
?
Hdefault_policy/gradients/default_policy/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs8default_policy/gradients/default_policy/mul_2_grad/Shape:default_policy/gradients/default_policy/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
6default_policy/gradients/default_policy/mul_2_grad/MulMulMdefault_policy/gradients/default_policy/Minimum_grad/tuple/control_dependencydefault_policy/Exp_1*
T0*#
_output_shapes
:?????????
?
6default_policy/gradients/default_policy/mul_2_grad/SumSum6default_policy/gradients/default_policy/mul_2_grad/MulHdefault_policy/gradients/default_policy/mul_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
:default_policy/gradients/default_policy/mul_2_grad/ReshapeReshape6default_policy/gradients/default_policy/mul_2_grad/Sum8default_policy/gradients/default_policy/mul_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/mul_2_grad/Mul_1Muldefault_policy/advantagesMdefault_policy/gradients/default_policy/Minimum_grad/tuple/control_dependency*
T0*#
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/mul_2_grad/Sum_1Sum8default_policy/gradients/default_policy/mul_2_grad/Mul_1Jdefault_policy/gradients/default_policy/mul_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
<default_policy/gradients/default_policy/mul_2_grad/Reshape_1Reshape8default_policy/gradients/default_policy/mul_2_grad/Sum_1:default_policy/gradients/default_policy/mul_2_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:?????????
?
Cdefault_policy/gradients/default_policy/mul_2_grad/tuple/group_depsNoOp;^default_policy/gradients/default_policy/mul_2_grad/Reshape=^default_policy/gradients/default_policy/mul_2_grad/Reshape_1
?
Kdefault_policy/gradients/default_policy/mul_2_grad/tuple/control_dependencyIdentity:default_policy/gradients/default_policy/mul_2_grad/ReshapeD^default_policy/gradients/default_policy/mul_2_grad/tuple/group_deps*
T0*M
_classC
A?loc:@default_policy/gradients/default_policy/mul_2_grad/Reshape*#
_output_shapes
:?????????
?
Mdefault_policy/gradients/default_policy/mul_2_grad/tuple/control_dependency_1Identity<default_policy/gradients/default_policy/mul_2_grad/Reshape_1D^default_policy/gradients/default_policy/mul_2_grad/tuple/group_deps*
T0*O
_classE
CAloc:@default_policy/gradients/default_policy/mul_2_grad/Reshape_1*#
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/mul_3_grad/ShapeShapedefault_policy/advantages*
T0*
_output_shapes
:*
out_type0
?
:default_policy/gradients/default_policy/mul_3_grad/Shape_1Shapedefault_policy/clip_by_value*
T0*
_output_shapes
:*
out_type0
?
Hdefault_policy/gradients/default_policy/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgs8default_policy/gradients/default_policy/mul_3_grad/Shape:default_policy/gradients/default_policy/mul_3_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
6default_policy/gradients/default_policy/mul_3_grad/MulMulOdefault_policy/gradients/default_policy/Minimum_grad/tuple/control_dependency_1default_policy/clip_by_value*
T0*#
_output_shapes
:?????????
?
6default_policy/gradients/default_policy/mul_3_grad/SumSum6default_policy/gradients/default_policy/mul_3_grad/MulHdefault_policy/gradients/default_policy/mul_3_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
:default_policy/gradients/default_policy/mul_3_grad/ReshapeReshape6default_policy/gradients/default_policy/mul_3_grad/Sum8default_policy/gradients/default_policy/mul_3_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/mul_3_grad/Mul_1Muldefault_policy/advantagesOdefault_policy/gradients/default_policy/Minimum_grad/tuple/control_dependency_1*
T0*#
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/mul_3_grad/Sum_1Sum8default_policy/gradients/default_policy/mul_3_grad/Mul_1Jdefault_policy/gradients/default_policy/mul_3_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
<default_policy/gradients/default_policy/mul_3_grad/Reshape_1Reshape8default_policy/gradients/default_policy/mul_3_grad/Sum_1:default_policy/gradients/default_policy/mul_3_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:?????????
?
Cdefault_policy/gradients/default_policy/mul_3_grad/tuple/group_depsNoOp;^default_policy/gradients/default_policy/mul_3_grad/Reshape=^default_policy/gradients/default_policy/mul_3_grad/Reshape_1
?
Kdefault_policy/gradients/default_policy/mul_3_grad/tuple/control_dependencyIdentity:default_policy/gradients/default_policy/mul_3_grad/ReshapeD^default_policy/gradients/default_policy/mul_3_grad/tuple/group_deps*
T0*M
_classC
A?loc:@default_policy/gradients/default_policy/mul_3_grad/Reshape*#
_output_shapes
:?????????
?
Mdefault_policy/gradients/default_policy/mul_3_grad/tuple/control_dependency_1Identity<default_policy/gradients/default_policy/mul_3_grad/Reshape_1D^default_policy/gradients/default_policy/mul_3_grad/tuple/group_deps*
T0*O
_classE
CAloc:@default_policy/gradients/default_policy/mul_3_grad/Reshape_1*#
_output_shapes
:?????????
?
6default_policy/gradients/default_policy/mul_grad/ShapeShapedefault_policy/truediv_4*
T0*
_output_shapes
:*
out_type0
?
8default_policy/gradients/default_policy/mul_grad/Shape_1Shapedefault_policy/add_1*
T0*
_output_shapes
:*
out_type0
?
Fdefault_policy/gradients/default_policy/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6default_policy/gradients/default_policy/mul_grad/Shape8default_policy/gradients/default_policy/mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
4default_policy/gradients/default_policy/mul_grad/MulMul>default_policy/gradients/default_policy/Sum_2_grad/BroadcastTodefault_policy/add_1*
T0*'
_output_shapes
:?????????
?
4default_policy/gradients/default_policy/mul_grad/SumSum4default_policy/gradients/default_policy/mul_grad/MulFdefault_policy/gradients/default_policy/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
8default_policy/gradients/default_policy/mul_grad/ReshapeReshape4default_policy/gradients/default_policy/mul_grad/Sum6default_policy/gradients/default_policy/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
6default_policy/gradients/default_policy/mul_grad/Mul_1Muldefault_policy/truediv_4>default_policy/gradients/default_policy/Sum_2_grad/BroadcastTo*
T0*'
_output_shapes
:?????????
?
6default_policy/gradients/default_policy/mul_grad/Sum_1Sum6default_policy/gradients/default_policy/mul_grad/Mul_1Hdefault_policy/gradients/default_policy/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
:default_policy/gradients/default_policy/mul_grad/Reshape_1Reshape6default_policy/gradients/default_policy/mul_grad/Sum_18default_policy/gradients/default_policy/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
?
Adefault_policy/gradients/default_policy/mul_grad/tuple/group_depsNoOp9^default_policy/gradients/default_policy/mul_grad/Reshape;^default_policy/gradients/default_policy/mul_grad/Reshape_1
?
Idefault_policy/gradients/default_policy/mul_grad/tuple/control_dependencyIdentity8default_policy/gradients/default_policy/mul_grad/ReshapeB^default_policy/gradients/default_policy/mul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@default_policy/gradients/default_policy/mul_grad/Reshape*'
_output_shapes
:?????????
?
Kdefault_policy/gradients/default_policy/mul_grad/tuple/control_dependency_1Identity:default_policy/gradients/default_policy/mul_grad/Reshape_1B^default_policy/gradients/default_policy/mul_grad/tuple/group_deps*
T0*M
_classC
A?loc:@default_policy/gradients/default_policy/mul_grad/Reshape_1*'
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/sub_7_grad/ShapeShapedefault_policy/Reshape_2*
T0*
_output_shapes
:*
out_type0
?
:default_policy/gradients/default_policy/sub_7_grad/Shape_1Shapedefault_policy/value_targets*
T0*
_output_shapes
:*
out_type0
?
Hdefault_policy/gradients/default_policy/sub_7_grad/BroadcastGradientArgsBroadcastGradientArgs8default_policy/gradients/default_policy/sub_7_grad/Shape:default_policy/gradients/default_policy/sub_7_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
6default_policy/gradients/default_policy/sub_7_grad/SumSum9default_policy/gradients/default_policy/Square_grad/Mul_1Hdefault_policy/gradients/default_policy/sub_7_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
:default_policy/gradients/default_policy/sub_7_grad/ReshapeReshape6default_policy/gradients/default_policy/sub_7_grad/Sum8default_policy/gradients/default_policy/sub_7_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
6default_policy/gradients/default_policy/sub_7_grad/NegNeg9default_policy/gradients/default_policy/Square_grad/Mul_1*
T0*#
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/sub_7_grad/Sum_1Sum6default_policy/gradients/default_policy/sub_7_grad/NegJdefault_policy/gradients/default_policy/sub_7_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
<default_policy/gradients/default_policy/sub_7_grad/Reshape_1Reshape8default_policy/gradients/default_policy/sub_7_grad/Sum_1:default_policy/gradients/default_policy/sub_7_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:?????????
?
Cdefault_policy/gradients/default_policy/sub_7_grad/tuple/group_depsNoOp;^default_policy/gradients/default_policy/sub_7_grad/Reshape=^default_policy/gradients/default_policy/sub_7_grad/Reshape_1
?
Kdefault_policy/gradients/default_policy/sub_7_grad/tuple/control_dependencyIdentity:default_policy/gradients/default_policy/sub_7_grad/ReshapeD^default_policy/gradients/default_policy/sub_7_grad/tuple/group_deps*
T0*M
_classC
A?loc:@default_policy/gradients/default_policy/sub_7_grad/Reshape*#
_output_shapes
:?????????
?
Mdefault_policy/gradients/default_policy/sub_7_grad/tuple/control_dependency_1Identity<default_policy/gradients/default_policy/sub_7_grad/Reshape_1D^default_policy/gradients/default_policy/sub_7_grad/tuple/group_deps*
T0*O
_classE
CAloc:@default_policy/gradients/default_policy/sub_7_grad/Reshape_1*#
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/sub_9_grad/ShapeShapedefault_policy/add_2*
T0*
_output_shapes
:*
out_type0
?
:default_policy/gradients/default_policy/sub_9_grad/Shape_1Shapedefault_policy/value_targets*
T0*
_output_shapes
:*
out_type0
?
Hdefault_policy/gradients/default_policy/sub_9_grad/BroadcastGradientArgsBroadcastGradientArgs8default_policy/gradients/default_policy/sub_9_grad/Shape:default_policy/gradients/default_policy/sub_9_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
6default_policy/gradients/default_policy/sub_9_grad/SumSum;default_policy/gradients/default_policy/Square_1_grad/Mul_1Hdefault_policy/gradients/default_policy/sub_9_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
:default_policy/gradients/default_policy/sub_9_grad/ReshapeReshape6default_policy/gradients/default_policy/sub_9_grad/Sum8default_policy/gradients/default_policy/sub_9_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
6default_policy/gradients/default_policy/sub_9_grad/NegNeg;default_policy/gradients/default_policy/Square_1_grad/Mul_1*
T0*#
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/sub_9_grad/Sum_1Sum6default_policy/gradients/default_policy/sub_9_grad/NegJdefault_policy/gradients/default_policy/sub_9_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
<default_policy/gradients/default_policy/sub_9_grad/Reshape_1Reshape8default_policy/gradients/default_policy/sub_9_grad/Sum_1:default_policy/gradients/default_policy/sub_9_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:?????????
?
Cdefault_policy/gradients/default_policy/sub_9_grad/tuple/group_depsNoOp;^default_policy/gradients/default_policy/sub_9_grad/Reshape=^default_policy/gradients/default_policy/sub_9_grad/Reshape_1
?
Kdefault_policy/gradients/default_policy/sub_9_grad/tuple/control_dependencyIdentity:default_policy/gradients/default_policy/sub_9_grad/ReshapeD^default_policy/gradients/default_policy/sub_9_grad/tuple/group_deps*
T0*M
_classC
A?loc:@default_policy/gradients/default_policy/sub_9_grad/Reshape*#
_output_shapes
:?????????
?
Mdefault_policy/gradients/default_policy/sub_9_grad/tuple/control_dependency_1Identity<default_policy/gradients/default_policy/sub_9_grad/Reshape_1D^default_policy/gradients/default_policy/sub_9_grad/tuple/group_deps*
T0*O
_classE
CAloc:@default_policy/gradients/default_policy/sub_9_grad/Reshape_1*#
_output_shapes
:?????????
?
=default_policy/gradients/default_policy/Log_2_grad/Reciprocal
Reciprocaldefault_policy/Sum_3L^default_policy/gradients/default_policy/sub_6_grad/tuple/control_dependency*
T0*'
_output_shapes
:?????????
?
6default_policy/gradients/default_policy/Log_2_grad/mulMulKdefault_policy/gradients/default_policy/sub_6_grad/tuple/control_dependency=default_policy/gradients/default_policy/Log_2_grad/Reciprocal*
T0*'
_output_shapes
:?????????
?
@default_policy/gradients/default_policy/clip_by_value_grad/ShapeShape$default_policy/clip_by_value/Minimum*
T0*
_output_shapes
:*
out_type0
?
Bdefault_policy/gradients/default_policy/clip_by_value_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
?
Edefault_policy/gradients/default_policy/clip_by_value_grad/zeros_like	ZerosLikeMdefault_policy/gradients/default_policy/mul_3_grad/tuple/control_dependency_1*
T0*#
_output_shapes
:?????????
?
Gdefault_policy/gradients/default_policy/clip_by_value_grad/GreaterEqualGreaterEqual$default_policy/clip_by_value/Minimumdefault_policy/clip_by_value/y*
T0*#
_output_shapes
:?????????
?
Pdefault_policy/gradients/default_policy/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgs@default_policy/gradients/default_policy/clip_by_value_grad/ShapeBdefault_policy/gradients/default_policy/clip_by_value_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
Cdefault_policy/gradients/default_policy/clip_by_value_grad/SelectV2SelectV2Gdefault_policy/gradients/default_policy/clip_by_value_grad/GreaterEqualMdefault_policy/gradients/default_policy/mul_3_grad/tuple/control_dependency_1Edefault_policy/gradients/default_policy/clip_by_value_grad/zeros_like*
T0*#
_output_shapes
:?????????
?
>default_policy/gradients/default_policy/clip_by_value_grad/SumSumCdefault_policy/gradients/default_policy/clip_by_value_grad/SelectV2Pdefault_policy/gradients/default_policy/clip_by_value_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
Bdefault_policy/gradients/default_policy/clip_by_value_grad/ReshapeReshape>default_policy/gradients/default_policy/clip_by_value_grad/Sum@default_policy/gradients/default_policy/clip_by_value_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
Edefault_policy/gradients/default_policy/clip_by_value_grad/SelectV2_1SelectV2Gdefault_policy/gradients/default_policy/clip_by_value_grad/GreaterEqualEdefault_policy/gradients/default_policy/clip_by_value_grad/zeros_likeMdefault_policy/gradients/default_policy/mul_3_grad/tuple/control_dependency_1*
T0*#
_output_shapes
:?????????
?
@default_policy/gradients/default_policy/clip_by_value_grad/Sum_1SumEdefault_policy/gradients/default_policy/clip_by_value_grad/SelectV2_1Rdefault_policy/gradients/default_policy/clip_by_value_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
Ddefault_policy/gradients/default_policy/clip_by_value_grad/Reshape_1Reshape@default_policy/gradients/default_policy/clip_by_value_grad/Sum_1Bdefault_policy/gradients/default_policy/clip_by_value_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
Kdefault_policy/gradients/default_policy/clip_by_value_grad/tuple/group_depsNoOpC^default_policy/gradients/default_policy/clip_by_value_grad/ReshapeE^default_policy/gradients/default_policy/clip_by_value_grad/Reshape_1
?
Sdefault_policy/gradients/default_policy/clip_by_value_grad/tuple/control_dependencyIdentityBdefault_policy/gradients/default_policy/clip_by_value_grad/ReshapeL^default_policy/gradients/default_policy/clip_by_value_grad/tuple/group_deps*
T0*U
_classK
IGloc:@default_policy/gradients/default_policy/clip_by_value_grad/Reshape*#
_output_shapes
:?????????
?
Udefault_policy/gradients/default_policy/clip_by_value_grad/tuple/control_dependency_1IdentityDdefault_policy/gradients/default_policy/clip_by_value_grad/Reshape_1L^default_policy/gradients/default_policy/clip_by_value_grad/tuple/group_deps*
T0*W
_classM
KIloc:@default_policy/gradients/default_policy/clip_by_value_grad/Reshape_1*
_output_shapes
: 
?
8default_policy/gradients/default_policy/add_1_grad/ShapeShapedefault_policy/sub_4*
T0*
_output_shapes
:*
out_type0
?
:default_policy/gradients/default_policy/add_1_grad/Shape_1Shapedefault_policy/Log_1*
T0*
_output_shapes
:*
out_type0
?
Hdefault_policy/gradients/default_policy/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs8default_policy/gradients/default_policy/add_1_grad/Shape:default_policy/gradients/default_policy/add_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
6default_policy/gradients/default_policy/add_1_grad/SumSumKdefault_policy/gradients/default_policy/mul_grad/tuple/control_dependency_1Hdefault_policy/gradients/default_policy/add_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
:default_policy/gradients/default_policy/add_1_grad/ReshapeReshape6default_policy/gradients/default_policy/add_1_grad/Sum8default_policy/gradients/default_policy/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/add_1_grad/Sum_1SumKdefault_policy/gradients/default_policy/mul_grad/tuple/control_dependency_1Jdefault_policy/gradients/default_policy/add_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
<default_policy/gradients/default_policy/add_1_grad/Reshape_1Reshape8default_policy/gradients/default_policy/add_1_grad/Sum_1:default_policy/gradients/default_policy/add_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
?
Cdefault_policy/gradients/default_policy/add_1_grad/tuple/group_depsNoOp;^default_policy/gradients/default_policy/add_1_grad/Reshape=^default_policy/gradients/default_policy/add_1_grad/Reshape_1
?
Kdefault_policy/gradients/default_policy/add_1_grad/tuple/control_dependencyIdentity:default_policy/gradients/default_policy/add_1_grad/ReshapeD^default_policy/gradients/default_policy/add_1_grad/tuple/group_deps*
T0*M
_classC
A?loc:@default_policy/gradients/default_policy/add_1_grad/Reshape*'
_output_shapes
:?????????
?
Mdefault_policy/gradients/default_policy/add_1_grad/tuple/control_dependency_1Identity<default_policy/gradients/default_policy/add_1_grad/Reshape_1D^default_policy/gradients/default_policy/add_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@default_policy/gradients/default_policy/add_1_grad/Reshape_1*'
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/add_2_grad/ShapeShapedefault_policy/vf_preds*
T0*
_output_shapes
:*
out_type0
?
:default_policy/gradients/default_policy/add_2_grad/Shape_1Shapedefault_policy/clip_by_value_1*
T0*
_output_shapes
:*
out_type0
?
Hdefault_policy/gradients/default_policy/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs8default_policy/gradients/default_policy/add_2_grad/Shape:default_policy/gradients/default_policy/add_2_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
6default_policy/gradients/default_policy/add_2_grad/SumSumKdefault_policy/gradients/default_policy/sub_9_grad/tuple/control_dependencyHdefault_policy/gradients/default_policy/add_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
:default_policy/gradients/default_policy/add_2_grad/ReshapeReshape6default_policy/gradients/default_policy/add_2_grad/Sum8default_policy/gradients/default_policy/add_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/add_2_grad/Sum_1SumKdefault_policy/gradients/default_policy/sub_9_grad/tuple/control_dependencyJdefault_policy/gradients/default_policy/add_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
<default_policy/gradients/default_policy/add_2_grad/Reshape_1Reshape8default_policy/gradients/default_policy/add_2_grad/Sum_1:default_policy/gradients/default_policy/add_2_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:?????????
?
Cdefault_policy/gradients/default_policy/add_2_grad/tuple/group_depsNoOp;^default_policy/gradients/default_policy/add_2_grad/Reshape=^default_policy/gradients/default_policy/add_2_grad/Reshape_1
?
Kdefault_policy/gradients/default_policy/add_2_grad/tuple/control_dependencyIdentity:default_policy/gradients/default_policy/add_2_grad/ReshapeD^default_policy/gradients/default_policy/add_2_grad/tuple/group_deps*
T0*M
_classC
A?loc:@default_policy/gradients/default_policy/add_2_grad/Reshape*#
_output_shapes
:?????????
?
Mdefault_policy/gradients/default_policy/add_2_grad/tuple/control_dependency_1Identity<default_policy/gradients/default_policy/add_2_grad/Reshape_1D^default_policy/gradients/default_policy/add_2_grad/tuple/group_deps*
T0*O
_classE
CAloc:@default_policy/gradients/default_policy/add_2_grad/Reshape_1*#
_output_shapes
:?????????
?
default_policy/gradients/AddNAddNQdefault_policy/gradients/default_policy/truediv_5_grad/tuple/control_dependency_16default_policy/gradients/default_policy/Log_2_grad/mul*
N*
T0*S
_classI
GEloc:@default_policy/gradients/default_policy/truediv_5_grad/Reshape_1*'
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/Sum_3_grad/ShapeShapedefault_policy/Exp_4*
T0*
_output_shapes
:*
out_type0
?
>default_policy/gradients/default_policy/Sum_3_grad/BroadcastToBroadcastTodefault_policy/gradients/AddN8default_policy/gradients/default_policy/Sum_3_grad/Shape*
T0*

Tidx0*'
_output_shapes
:?????????
?
Hdefault_policy/gradients/default_policy/clip_by_value/Minimum_grad/ShapeShapedefault_policy/Exp_1*
T0*
_output_shapes
:*
out_type0
?
Jdefault_policy/gradients/default_policy/clip_by_value/Minimum_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
?
Mdefault_policy/gradients/default_policy/clip_by_value/Minimum_grad/zeros_like	ZerosLikeSdefault_policy/gradients/default_policy/clip_by_value_grad/tuple/control_dependency*
T0*#
_output_shapes
:?????????
?
Ldefault_policy/gradients/default_policy/clip_by_value/Minimum_grad/LessEqual	LessEqualdefault_policy/Exp_1&default_policy/clip_by_value/Minimum/y*
T0*#
_output_shapes
:?????????
?
Xdefault_policy/gradients/default_policy/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsHdefault_policy/gradients/default_policy/clip_by_value/Minimum_grad/ShapeJdefault_policy/gradients/default_policy/clip_by_value/Minimum_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
Kdefault_policy/gradients/default_policy/clip_by_value/Minimum_grad/SelectV2SelectV2Ldefault_policy/gradients/default_policy/clip_by_value/Minimum_grad/LessEqualSdefault_policy/gradients/default_policy/clip_by_value_grad/tuple/control_dependencyMdefault_policy/gradients/default_policy/clip_by_value/Minimum_grad/zeros_like*
T0*#
_output_shapes
:?????????
?
Fdefault_policy/gradients/default_policy/clip_by_value/Minimum_grad/SumSumKdefault_policy/gradients/default_policy/clip_by_value/Minimum_grad/SelectV2Xdefault_policy/gradients/default_policy/clip_by_value/Minimum_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
Jdefault_policy/gradients/default_policy/clip_by_value/Minimum_grad/ReshapeReshapeFdefault_policy/gradients/default_policy/clip_by_value/Minimum_grad/SumHdefault_policy/gradients/default_policy/clip_by_value/Minimum_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
Mdefault_policy/gradients/default_policy/clip_by_value/Minimum_grad/SelectV2_1SelectV2Ldefault_policy/gradients/default_policy/clip_by_value/Minimum_grad/LessEqualMdefault_policy/gradients/default_policy/clip_by_value/Minimum_grad/zeros_likeSdefault_policy/gradients/default_policy/clip_by_value_grad/tuple/control_dependency*
T0*#
_output_shapes
:?????????
?
Hdefault_policy/gradients/default_policy/clip_by_value/Minimum_grad/Sum_1SumMdefault_policy/gradients/default_policy/clip_by_value/Minimum_grad/SelectV2_1Zdefault_policy/gradients/default_policy/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
Ldefault_policy/gradients/default_policy/clip_by_value/Minimum_grad/Reshape_1ReshapeHdefault_policy/gradients/default_policy/clip_by_value/Minimum_grad/Sum_1Jdefault_policy/gradients/default_policy/clip_by_value/Minimum_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
Sdefault_policy/gradients/default_policy/clip_by_value/Minimum_grad/tuple/group_depsNoOpK^default_policy/gradients/default_policy/clip_by_value/Minimum_grad/ReshapeM^default_policy/gradients/default_policy/clip_by_value/Minimum_grad/Reshape_1
?
[default_policy/gradients/default_policy/clip_by_value/Minimum_grad/tuple/control_dependencyIdentityJdefault_policy/gradients/default_policy/clip_by_value/Minimum_grad/ReshapeT^default_policy/gradients/default_policy/clip_by_value/Minimum_grad/tuple/group_deps*
T0*]
_classS
QOloc:@default_policy/gradients/default_policy/clip_by_value/Minimum_grad/Reshape*#
_output_shapes
:?????????
?
]default_policy/gradients/default_policy/clip_by_value/Minimum_grad/tuple/control_dependency_1IdentityLdefault_policy/gradients/default_policy/clip_by_value/Minimum_grad/Reshape_1T^default_policy/gradients/default_policy/clip_by_value/Minimum_grad/tuple/group_deps*
T0*_
_classU
SQloc:@default_policy/gradients/default_policy/clip_by_value/Minimum_grad/Reshape_1*
_output_shapes
: 
?
8default_policy/gradients/default_policy/sub_4_grad/ShapeShapedefault_policy/sub_3*
T0*
_output_shapes
:*
out_type0
?
:default_policy/gradients/default_policy/sub_4_grad/Shape_1Shapedefault_policy/sub_2*
T0*
_output_shapes
:*
out_type0
?
Hdefault_policy/gradients/default_policy/sub_4_grad/BroadcastGradientArgsBroadcastGradientArgs8default_policy/gradients/default_policy/sub_4_grad/Shape:default_policy/gradients/default_policy/sub_4_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
6default_policy/gradients/default_policy/sub_4_grad/SumSumKdefault_policy/gradients/default_policy/add_1_grad/tuple/control_dependencyHdefault_policy/gradients/default_policy/sub_4_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
:default_policy/gradients/default_policy/sub_4_grad/ReshapeReshape6default_policy/gradients/default_policy/sub_4_grad/Sum8default_policy/gradients/default_policy/sub_4_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
6default_policy/gradients/default_policy/sub_4_grad/NegNegKdefault_policy/gradients/default_policy/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/sub_4_grad/Sum_1Sum6default_policy/gradients/default_policy/sub_4_grad/NegJdefault_policy/gradients/default_policy/sub_4_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
<default_policy/gradients/default_policy/sub_4_grad/Reshape_1Reshape8default_policy/gradients/default_policy/sub_4_grad/Sum_1:default_policy/gradients/default_policy/sub_4_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
?
Cdefault_policy/gradients/default_policy/sub_4_grad/tuple/group_depsNoOp;^default_policy/gradients/default_policy/sub_4_grad/Reshape=^default_policy/gradients/default_policy/sub_4_grad/Reshape_1
?
Kdefault_policy/gradients/default_policy/sub_4_grad/tuple/control_dependencyIdentity:default_policy/gradients/default_policy/sub_4_grad/ReshapeD^default_policy/gradients/default_policy/sub_4_grad/tuple/group_deps*
T0*M
_classC
A?loc:@default_policy/gradients/default_policy/sub_4_grad/Reshape*'
_output_shapes
:?????????
?
Mdefault_policy/gradients/default_policy/sub_4_grad/tuple/control_dependency_1Identity<default_policy/gradients/default_policy/sub_4_grad/Reshape_1D^default_policy/gradients/default_policy/sub_4_grad/tuple/group_deps*
T0*O
_classE
CAloc:@default_policy/gradients/default_policy/sub_4_grad/Reshape_1*'
_output_shapes
:?????????
?
=default_policy/gradients/default_policy/Log_1_grad/Reciprocal
Reciprocaldefault_policy/Sum_1N^default_policy/gradients/default_policy/add_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:?????????
?
6default_policy/gradients/default_policy/Log_1_grad/mulMulMdefault_policy/gradients/default_policy/add_1_grad/tuple/control_dependency_1=default_policy/gradients/default_policy/Log_1_grad/Reciprocal*
T0*'
_output_shapes
:?????????
?
Bdefault_policy/gradients/default_policy/clip_by_value_1_grad/ShapeShape&default_policy/clip_by_value_1/Minimum*
T0*
_output_shapes
:*
out_type0
?
Ddefault_policy/gradients/default_policy/clip_by_value_1_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
?
Gdefault_policy/gradients/default_policy/clip_by_value_1_grad/zeros_like	ZerosLikeMdefault_policy/gradients/default_policy/add_2_grad/tuple/control_dependency_1*
T0*#
_output_shapes
:?????????
?
Idefault_policy/gradients/default_policy/clip_by_value_1_grad/GreaterEqualGreaterEqual&default_policy/clip_by_value_1/Minimum default_policy/clip_by_value_1/y*
T0*#
_output_shapes
:?????????
?
Rdefault_policy/gradients/default_policy/clip_by_value_1_grad/BroadcastGradientArgsBroadcastGradientArgsBdefault_policy/gradients/default_policy/clip_by_value_1_grad/ShapeDdefault_policy/gradients/default_policy/clip_by_value_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
Edefault_policy/gradients/default_policy/clip_by_value_1_grad/SelectV2SelectV2Idefault_policy/gradients/default_policy/clip_by_value_1_grad/GreaterEqualMdefault_policy/gradients/default_policy/add_2_grad/tuple/control_dependency_1Gdefault_policy/gradients/default_policy/clip_by_value_1_grad/zeros_like*
T0*#
_output_shapes
:?????????
?
@default_policy/gradients/default_policy/clip_by_value_1_grad/SumSumEdefault_policy/gradients/default_policy/clip_by_value_1_grad/SelectV2Rdefault_policy/gradients/default_policy/clip_by_value_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
Ddefault_policy/gradients/default_policy/clip_by_value_1_grad/ReshapeReshape@default_policy/gradients/default_policy/clip_by_value_1_grad/SumBdefault_policy/gradients/default_policy/clip_by_value_1_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
Gdefault_policy/gradients/default_policy/clip_by_value_1_grad/SelectV2_1SelectV2Idefault_policy/gradients/default_policy/clip_by_value_1_grad/GreaterEqualGdefault_policy/gradients/default_policy/clip_by_value_1_grad/zeros_likeMdefault_policy/gradients/default_policy/add_2_grad/tuple/control_dependency_1*
T0*#
_output_shapes
:?????????
?
Bdefault_policy/gradients/default_policy/clip_by_value_1_grad/Sum_1SumGdefault_policy/gradients/default_policy/clip_by_value_1_grad/SelectV2_1Tdefault_policy/gradients/default_policy/clip_by_value_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
Fdefault_policy/gradients/default_policy/clip_by_value_1_grad/Reshape_1ReshapeBdefault_policy/gradients/default_policy/clip_by_value_1_grad/Sum_1Ddefault_policy/gradients/default_policy/clip_by_value_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
Mdefault_policy/gradients/default_policy/clip_by_value_1_grad/tuple/group_depsNoOpE^default_policy/gradients/default_policy/clip_by_value_1_grad/ReshapeG^default_policy/gradients/default_policy/clip_by_value_1_grad/Reshape_1
?
Udefault_policy/gradients/default_policy/clip_by_value_1_grad/tuple/control_dependencyIdentityDdefault_policy/gradients/default_policy/clip_by_value_1_grad/ReshapeN^default_policy/gradients/default_policy/clip_by_value_1_grad/tuple/group_deps*
T0*W
_classM
KIloc:@default_policy/gradients/default_policy/clip_by_value_1_grad/Reshape*#
_output_shapes
:?????????
?
Wdefault_policy/gradients/default_policy/clip_by_value_1_grad/tuple/control_dependency_1IdentityFdefault_policy/gradients/default_policy/clip_by_value_1_grad/Reshape_1N^default_policy/gradients/default_policy/clip_by_value_1_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@default_policy/gradients/default_policy/clip_by_value_1_grad/Reshape_1*
_output_shapes
: 
?
default_policy/gradients/AddN_1AddNOdefault_policy/gradients/default_policy/truediv_5_grad/tuple/control_dependency>default_policy/gradients/default_policy/Sum_3_grad/BroadcastTo*
N*
T0*Q
_classG
ECloc:@default_policy/gradients/default_policy/truediv_5_grad/Reshape*'
_output_shapes
:?????????
?
6default_policy/gradients/default_policy/Exp_4_grad/mulMuldefault_policy/gradients/AddN_1default_policy/Exp_4*
T0*'
_output_shapes
:?????????
?
default_policy/gradients/AddN_2AddNMdefault_policy/gradients/default_policy/mul_2_grad/tuple/control_dependency_1[default_policy/gradients/default_policy/clip_by_value/Minimum_grad/tuple/control_dependency*
N*
T0*O
_classE
CAloc:@default_policy/gradients/default_policy/mul_2_grad/Reshape_1*#
_output_shapes
:?????????
?
6default_policy/gradients/default_policy/Exp_1_grad/mulMuldefault_policy/gradients/AddN_2default_policy/Exp_1*
T0*#
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/Sum_1_grad/ShapeShapedefault_policy/Exp_3*
T0*
_output_shapes
:*
out_type0
?
>default_policy/gradients/default_policy/Sum_1_grad/BroadcastToBroadcastTo6default_policy/gradients/default_policy/Log_1_grad/mul8default_policy/gradients/default_policy/Sum_1_grad/Shape*
T0*

Tidx0*'
_output_shapes
:?????????
?
Jdefault_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/ShapeShapedefault_policy/sub_8*
T0*
_output_shapes
:*
out_type0
?
Ldefault_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
?
Odefault_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/zeros_like	ZerosLikeUdefault_policy/gradients/default_policy/clip_by_value_1_grad/tuple/control_dependency*
T0*#
_output_shapes
:?????????
?
Ndefault_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/LessEqual	LessEqualdefault_policy/sub_8(default_policy/clip_by_value_1/Minimum/y*
T0*#
_output_shapes
:?????????
?
Zdefault_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsJdefault_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/ShapeLdefault_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
Mdefault_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/SelectV2SelectV2Ndefault_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/LessEqualUdefault_policy/gradients/default_policy/clip_by_value_1_grad/tuple/control_dependencyOdefault_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/zeros_like*
T0*#
_output_shapes
:?????????
?
Hdefault_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/SumSumMdefault_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/SelectV2Zdefault_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
Ldefault_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/ReshapeReshapeHdefault_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/SumJdefault_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
Odefault_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/SelectV2_1SelectV2Ndefault_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/LessEqualOdefault_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/zeros_likeUdefault_policy/gradients/default_policy/clip_by_value_1_grad/tuple/control_dependency*
T0*#
_output_shapes
:?????????
?
Jdefault_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/Sum_1SumOdefault_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/SelectV2_1\default_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
Ndefault_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/Reshape_1ReshapeJdefault_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/Sum_1Ldefault_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
Udefault_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/tuple/group_depsNoOpM^default_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/ReshapeO^default_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/Reshape_1
?
]default_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/tuple/control_dependencyIdentityLdefault_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/ReshapeV^default_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/tuple/group_deps*
T0*_
_classU
SQloc:@default_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/Reshape*#
_output_shapes
:?????????
?
_default_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/tuple/control_dependency_1IdentityNdefault_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/Reshape_1V^default_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/tuple/group_deps*
T0*a
_classW
USloc:@default_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/Reshape_1*
_output_shapes
: 
?
default_policy/gradients/AddN_3AddNMdefault_policy/gradients/default_policy/sub_6_grad/tuple/control_dependency_16default_policy/gradients/default_policy/Exp_4_grad/mul*
N*
T0*O
_classE
CAloc:@default_policy/gradients/default_policy/sub_6_grad/Reshape_1*'
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/sub_5_grad/ShapeShapedefault_policy/truediv_2*
T0*
_output_shapes
:*
out_type0
?
:default_policy/gradients/default_policy/sub_5_grad/Shape_1Shapedefault_policy/Max_2*
T0*
_output_shapes
:*
out_type0
?
Hdefault_policy/gradients/default_policy/sub_5_grad/BroadcastGradientArgsBroadcastGradientArgs8default_policy/gradients/default_policy/sub_5_grad/Shape:default_policy/gradients/default_policy/sub_5_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
6default_policy/gradients/default_policy/sub_5_grad/SumSumdefault_policy/gradients/AddN_3Hdefault_policy/gradients/default_policy/sub_5_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
:default_policy/gradients/default_policy/sub_5_grad/ReshapeReshape6default_policy/gradients/default_policy/sub_5_grad/Sum8default_policy/gradients/default_policy/sub_5_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
6default_policy/gradients/default_policy/sub_5_grad/NegNegdefault_policy/gradients/AddN_3*
T0*'
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/sub_5_grad/Sum_1Sum6default_policy/gradients/default_policy/sub_5_grad/NegJdefault_policy/gradients/default_policy/sub_5_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
<default_policy/gradients/default_policy/sub_5_grad/Reshape_1Reshape8default_policy/gradients/default_policy/sub_5_grad/Sum_1:default_policy/gradients/default_policy/sub_5_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
?
Cdefault_policy/gradients/default_policy/sub_5_grad/tuple/group_depsNoOp;^default_policy/gradients/default_policy/sub_5_grad/Reshape=^default_policy/gradients/default_policy/sub_5_grad/Reshape_1
?
Kdefault_policy/gradients/default_policy/sub_5_grad/tuple/control_dependencyIdentity:default_policy/gradients/default_policy/sub_5_grad/ReshapeD^default_policy/gradients/default_policy/sub_5_grad/tuple/group_deps*
T0*M
_classC
A?loc:@default_policy/gradients/default_policy/sub_5_grad/Reshape*'
_output_shapes
:?????????
?
Mdefault_policy/gradients/default_policy/sub_5_grad/tuple/control_dependency_1Identity<default_policy/gradients/default_policy/sub_5_grad/Reshape_1D^default_policy/gradients/default_policy/sub_5_grad/tuple/group_deps*
T0*O
_classE
CAloc:@default_policy/gradients/default_policy/sub_5_grad/Reshape_1*'
_output_shapes
:?????????
?
6default_policy/gradients/default_policy/sub_grad/ShapeShapedefault_policy/Neg_5*
T0*
_output_shapes
:*
out_type0
?
8default_policy/gradients/default_policy/sub_grad/Shape_1Shapedefault_policy/action_logp*
T0*
_output_shapes
:*
out_type0
?
Fdefault_policy/gradients/default_policy/sub_grad/BroadcastGradientArgsBroadcastGradientArgs6default_policy/gradients/default_policy/sub_grad/Shape8default_policy/gradients/default_policy/sub_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
4default_policy/gradients/default_policy/sub_grad/SumSum6default_policy/gradients/default_policy/Exp_1_grad/mulFdefault_policy/gradients/default_policy/sub_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
8default_policy/gradients/default_policy/sub_grad/ReshapeReshape4default_policy/gradients/default_policy/sub_grad/Sum6default_policy/gradients/default_policy/sub_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
4default_policy/gradients/default_policy/sub_grad/NegNeg6default_policy/gradients/default_policy/Exp_1_grad/mul*
T0*#
_output_shapes
:?????????
?
6default_policy/gradients/default_policy/sub_grad/Sum_1Sum4default_policy/gradients/default_policy/sub_grad/NegHdefault_policy/gradients/default_policy/sub_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
:default_policy/gradients/default_policy/sub_grad/Reshape_1Reshape6default_policy/gradients/default_policy/sub_grad/Sum_18default_policy/gradients/default_policy/sub_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:?????????
?
Adefault_policy/gradients/default_policy/sub_grad/tuple/group_depsNoOp9^default_policy/gradients/default_policy/sub_grad/Reshape;^default_policy/gradients/default_policy/sub_grad/Reshape_1
?
Idefault_policy/gradients/default_policy/sub_grad/tuple/control_dependencyIdentity8default_policy/gradients/default_policy/sub_grad/ReshapeB^default_policy/gradients/default_policy/sub_grad/tuple/group_deps*
T0*K
_classA
?=loc:@default_policy/gradients/default_policy/sub_grad/Reshape*#
_output_shapes
:?????????
?
Kdefault_policy/gradients/default_policy/sub_grad/tuple/control_dependency_1Identity:default_policy/gradients/default_policy/sub_grad/Reshape_1B^default_policy/gradients/default_policy/sub_grad/tuple/group_deps*
T0*M
_classC
A?loc:@default_policy/gradients/default_policy/sub_grad/Reshape_1*#
_output_shapes
:?????????
?
6default_policy/gradients/default_policy/Exp_3_grad/mulMul>default_policy/gradients/default_policy/Sum_1_grad/BroadcastTodefault_policy/Exp_3*
T0*'
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/sub_8_grad/ShapeShapedefault_policy/Reshape_2*
T0*
_output_shapes
:*
out_type0
?
:default_policy/gradients/default_policy/sub_8_grad/Shape_1Shapedefault_policy/vf_preds*
T0*
_output_shapes
:*
out_type0
?
Hdefault_policy/gradients/default_policy/sub_8_grad/BroadcastGradientArgsBroadcastGradientArgs8default_policy/gradients/default_policy/sub_8_grad/Shape:default_policy/gradients/default_policy/sub_8_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
6default_policy/gradients/default_policy/sub_8_grad/SumSum]default_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/tuple/control_dependencyHdefault_policy/gradients/default_policy/sub_8_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
:default_policy/gradients/default_policy/sub_8_grad/ReshapeReshape6default_policy/gradients/default_policy/sub_8_grad/Sum8default_policy/gradients/default_policy/sub_8_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
6default_policy/gradients/default_policy/sub_8_grad/NegNeg]default_policy/gradients/default_policy/clip_by_value_1/Minimum_grad/tuple/control_dependency*
T0*#
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/sub_8_grad/Sum_1Sum6default_policy/gradients/default_policy/sub_8_grad/NegJdefault_policy/gradients/default_policy/sub_8_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
<default_policy/gradients/default_policy/sub_8_grad/Reshape_1Reshape8default_policy/gradients/default_policy/sub_8_grad/Sum_1:default_policy/gradients/default_policy/sub_8_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:?????????
?
Cdefault_policy/gradients/default_policy/sub_8_grad/tuple/group_depsNoOp;^default_policy/gradients/default_policy/sub_8_grad/Reshape=^default_policy/gradients/default_policy/sub_8_grad/Reshape_1
?
Kdefault_policy/gradients/default_policy/sub_8_grad/tuple/control_dependencyIdentity:default_policy/gradients/default_policy/sub_8_grad/ReshapeD^default_policy/gradients/default_policy/sub_8_grad/tuple/group_deps*
T0*M
_classC
A?loc:@default_policy/gradients/default_policy/sub_8_grad/Reshape*#
_output_shapes
:?????????
?
Mdefault_policy/gradients/default_policy/sub_8_grad/tuple/control_dependency_1Identity<default_policy/gradients/default_policy/sub_8_grad/Reshape_1D^default_policy/gradients/default_policy/sub_8_grad/tuple/group_deps*
T0*O
_classE
CAloc:@default_policy/gradients/default_policy/sub_8_grad/Reshape_1*#
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/Max_2_grad/ShapeShapedefault_policy/truediv_2*
T0*
_output_shapes
:*
out_type0
?
:default_policy/gradients/default_policy/Max_2_grad/Shape_1Shapedefault_policy/Max_2*
T0*
_output_shapes
:*
out_type0
?
8default_policy/gradients/default_policy/Max_2_grad/EqualEqualdefault_policy/Max_2default_policy/truediv_2*
T0*'
_output_shapes
:?????????*
incompatible_shape_error(
?
7default_policy/gradients/default_policy/Max_2_grad/CastCast8default_policy/gradients/default_policy/Max_2_grad/Equal*

DstT0*

SrcT0
*
Truncate( *'
_output_shapes
:?????????
?
6default_policy/gradients/default_policy/Max_2_grad/SumSum7default_policy/gradients/default_policy/Max_2_grad/Cast&default_policy/Max_2/reduction_indices*
T0*

Tidx0*#
_output_shapes
:?????????*
	keep_dims( 
?
:default_policy/gradients/default_policy/Max_2_grad/ReshapeReshape6default_policy/gradients/default_policy/Max_2_grad/Sum:default_policy/gradients/default_policy/Max_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
?
:default_policy/gradients/default_policy/Max_2_grad/truedivRealDiv7default_policy/gradients/default_policy/Max_2_grad/Cast:default_policy/gradients/default_policy/Max_2_grad/Reshape*
T0*'
_output_shapes
:?????????
?
6default_policy/gradients/default_policy/Max_2_grad/mulMul:default_policy/gradients/default_policy/Max_2_grad/truedivMdefault_policy/gradients/default_policy/sub_5_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:?????????
?
6default_policy/gradients/default_policy/Neg_5_grad/NegNegIdefault_policy/gradients/default_policy/sub_grad/tuple/control_dependency*
T0*#
_output_shapes
:?????????
?
default_policy/gradients/AddN_4AddNMdefault_policy/gradients/default_policy/sub_4_grad/tuple/control_dependency_16default_policy/gradients/default_policy/Exp_3_grad/mul*
N*
T0*O
_classE
CAloc:@default_policy/gradients/default_policy/sub_4_grad/Reshape_1*'
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/sub_2_grad/ShapeShapedefault_policy/truediv_2*
T0*
_output_shapes
:*
out_type0
?
:default_policy/gradients/default_policy/sub_2_grad/Shape_1Shapedefault_policy/Max_1*
T0*
_output_shapes
:*
out_type0
?
Hdefault_policy/gradients/default_policy/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgs8default_policy/gradients/default_policy/sub_2_grad/Shape:default_policy/gradients/default_policy/sub_2_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
6default_policy/gradients/default_policy/sub_2_grad/SumSumdefault_policy/gradients/AddN_4Hdefault_policy/gradients/default_policy/sub_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
:default_policy/gradients/default_policy/sub_2_grad/ReshapeReshape6default_policy/gradients/default_policy/sub_2_grad/Sum8default_policy/gradients/default_policy/sub_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
6default_policy/gradients/default_policy/sub_2_grad/NegNegdefault_policy/gradients/AddN_4*
T0*'
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/sub_2_grad/Sum_1Sum6default_policy/gradients/default_policy/sub_2_grad/NegJdefault_policy/gradients/default_policy/sub_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
<default_policy/gradients/default_policy/sub_2_grad/Reshape_1Reshape8default_policy/gradients/default_policy/sub_2_grad/Sum_1:default_policy/gradients/default_policy/sub_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
?
Cdefault_policy/gradients/default_policy/sub_2_grad/tuple/group_depsNoOp;^default_policy/gradients/default_policy/sub_2_grad/Reshape=^default_policy/gradients/default_policy/sub_2_grad/Reshape_1
?
Kdefault_policy/gradients/default_policy/sub_2_grad/tuple/control_dependencyIdentity:default_policy/gradients/default_policy/sub_2_grad/ReshapeD^default_policy/gradients/default_policy/sub_2_grad/tuple/group_deps*
T0*M
_classC
A?loc:@default_policy/gradients/default_policy/sub_2_grad/Reshape*'
_output_shapes
:?????????
?
Mdefault_policy/gradients/default_policy/sub_2_grad/tuple/control_dependency_1Identity<default_policy/gradients/default_policy/sub_2_grad/Reshape_1D^default_policy/gradients/default_policy/sub_2_grad/tuple/group_deps*
T0*O
_classE
CAloc:@default_policy/gradients/default_policy/sub_2_grad/Reshape_1*'
_output_shapes
:?????????
?
default_policy/gradients/AddN_5AddNKdefault_policy/gradients/default_policy/sub_7_grad/tuple/control_dependencyKdefault_policy/gradients/default_policy/sub_8_grad/tuple/control_dependency*
N*
T0*M
_classC
A?loc:@default_policy/gradients/default_policy/sub_7_grad/Reshape*#
_output_shapes
:?????????
?
<default_policy/gradients/default_policy/Reshape_2_grad/ShapeShape%default_policy/model_2/lambda/Squeeze*
T0*
_output_shapes
:*
out_type0
?
>default_policy/gradients/default_policy/Reshape_2_grad/ReshapeReshapedefault_policy/gradients/AddN_5<default_policy/gradients/default_policy/Reshape_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
#default_policy/gradients/zeros_like	ZerosLikeZdefault_policy/SparseSoftmaxCrossEntropyWithLogits_5/SparseSoftmaxCrossEntropyWithLogits:1*
T0*'
_output_shapes
:?????????
?
?default_policy/gradients/default_policy/SparseSoftmaxCrossEntropyWithLogits_5/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????
?
?default_policy/gradients/default_policy/SparseSoftmaxCrossEntropyWithLogits_5/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims6default_policy/gradients/default_policy/Neg_5_grad/Neg?default_policy/gradients/default_policy/SparseSoftmaxCrossEntropyWithLogits_5/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
T0*

Tdim0*'
_output_shapes
:?????????
?
zdefault_policy/gradients/default_policy/SparseSoftmaxCrossEntropyWithLogits_5/SparseSoftmaxCrossEntropyWithLogits_grad/mulMul?default_policy/gradients/default_policy/SparseSoftmaxCrossEntropyWithLogits_5/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimsZdefault_policy/SparseSoftmaxCrossEntropyWithLogits_5/SparseSoftmaxCrossEntropyWithLogits:1*
T0*'
_output_shapes
:?????????
?
8default_policy/gradients/default_policy/Max_1_grad/ShapeShapedefault_policy/truediv_2*
T0*
_output_shapes
:*
out_type0
?
:default_policy/gradients/default_policy/Max_1_grad/Shape_1Shapedefault_policy/Max_1*
T0*
_output_shapes
:*
out_type0
?
8default_policy/gradients/default_policy/Max_1_grad/EqualEqualdefault_policy/Max_1default_policy/truediv_2*
T0*'
_output_shapes
:?????????*
incompatible_shape_error(
?
7default_policy/gradients/default_policy/Max_1_grad/CastCast8default_policy/gradients/default_policy/Max_1_grad/Equal*

DstT0*

SrcT0
*
Truncate( *'
_output_shapes
:?????????
?
6default_policy/gradients/default_policy/Max_1_grad/SumSum7default_policy/gradients/default_policy/Max_1_grad/Cast&default_policy/Max_1/reduction_indices*
T0*

Tidx0*#
_output_shapes
:?????????*
	keep_dims( 
?
:default_policy/gradients/default_policy/Max_1_grad/ReshapeReshape6default_policy/gradients/default_policy/Max_1_grad/Sum:default_policy/gradients/default_policy/Max_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
?
:default_policy/gradients/default_policy/Max_1_grad/truedivRealDiv7default_policy/gradients/default_policy/Max_1_grad/Cast:default_policy/gradients/default_policy/Max_1_grad/Reshape*
T0*'
_output_shapes
:?????????
?
6default_policy/gradients/default_policy/Max_1_grad/mulMul:default_policy/gradients/default_policy/Max_1_grad/truedivMdefault_policy/gradients/default_policy/sub_2_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:?????????
?
Idefault_policy/gradients/default_policy/model_2/lambda/Squeeze_grad/ShapeShape-default_policy/model_2/conv_value_out/BiasAdd*
T0*
_output_shapes
:*
out_type0
?
Kdefault_policy/gradients/default_policy/model_2/lambda/Squeeze_grad/ReshapeReshape>default_policy/gradients/default_policy/Reshape_2_grad/ReshapeIdefault_policy/gradients/default_policy/model_2/lambda/Squeeze_grad/Shape*
T0*
Tshape0*/
_output_shapes
:?????????
?
default_policy/gradients/AddN_6AddNKdefault_policy/gradients/default_policy/sub_5_grad/tuple/control_dependency6default_policy/gradients/default_policy/Max_2_grad/mulKdefault_policy/gradients/default_policy/sub_2_grad/tuple/control_dependencyzdefault_policy/gradients/default_policy/SparseSoftmaxCrossEntropyWithLogits_5/SparseSoftmaxCrossEntropyWithLogits_grad/mul6default_policy/gradients/default_policy/Max_1_grad/mul*
N*
T0*M
_classC
A?loc:@default_policy/gradients/default_policy/sub_5_grad/Reshape*'
_output_shapes
:?????????
?
<default_policy/gradients/default_policy/truediv_2_grad/ShapeShapedefault_policy/Squeeze_4*
T0*
_output_shapes
:*
out_type0
?
>default_policy/gradients/default_policy/truediv_2_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
?
Ldefault_policy/gradients/default_policy/truediv_2_grad/BroadcastGradientArgsBroadcastGradientArgs<default_policy/gradients/default_policy/truediv_2_grad/Shape>default_policy/gradients/default_policy/truediv_2_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
>default_policy/gradients/default_policy/truediv_2_grad/RealDivRealDivdefault_policy/gradients/AddN_6default_policy/truediv_2/y*
T0*'
_output_shapes
:?????????
?
:default_policy/gradients/default_policy/truediv_2_grad/SumSum>default_policy/gradients/default_policy/truediv_2_grad/RealDivLdefault_policy/gradients/default_policy/truediv_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
>default_policy/gradients/default_policy/truediv_2_grad/ReshapeReshape:default_policy/gradients/default_policy/truediv_2_grad/Sum<default_policy/gradients/default_policy/truediv_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
:default_policy/gradients/default_policy/truediv_2_grad/NegNegdefault_policy/Squeeze_4*
T0*'
_output_shapes
:?????????
?
@default_policy/gradients/default_policy/truediv_2_grad/RealDiv_1RealDiv:default_policy/gradients/default_policy/truediv_2_grad/Negdefault_policy/truediv_2/y*
T0*'
_output_shapes
:?????????
?
@default_policy/gradients/default_policy/truediv_2_grad/RealDiv_2RealDiv@default_policy/gradients/default_policy/truediv_2_grad/RealDiv_1default_policy/truediv_2/y*
T0*'
_output_shapes
:?????????
?
:default_policy/gradients/default_policy/truediv_2_grad/mulMuldefault_policy/gradients/AddN_6@default_policy/gradients/default_policy/truediv_2_grad/RealDiv_2*
T0*'
_output_shapes
:?????????
?
<default_policy/gradients/default_policy/truediv_2_grad/Sum_1Sum:default_policy/gradients/default_policy/truediv_2_grad/mulNdefault_policy/gradients/default_policy/truediv_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
@default_policy/gradients/default_policy/truediv_2_grad/Reshape_1Reshape<default_policy/gradients/default_policy/truediv_2_grad/Sum_1>default_policy/gradients/default_policy/truediv_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
?
Gdefault_policy/gradients/default_policy/truediv_2_grad/tuple/group_depsNoOp?^default_policy/gradients/default_policy/truediv_2_grad/ReshapeA^default_policy/gradients/default_policy/truediv_2_grad/Reshape_1
?
Odefault_policy/gradients/default_policy/truediv_2_grad/tuple/control_dependencyIdentity>default_policy/gradients/default_policy/truediv_2_grad/ReshapeH^default_policy/gradients/default_policy/truediv_2_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@default_policy/gradients/default_policy/truediv_2_grad/Reshape*'
_output_shapes
:?????????
?
Qdefault_policy/gradients/default_policy/truediv_2_grad/tuple/control_dependency_1Identity@default_policy/gradients/default_policy/truediv_2_grad/Reshape_1H^default_policy/gradients/default_policy/truediv_2_grad/tuple/group_deps*
T0*S
_classI
GEloc:@default_policy/gradients/default_policy/truediv_2_grad/Reshape_1*
_output_shapes
: 
?
Wdefault_policy/gradients/default_policy/model_2/conv_value_out/BiasAdd_grad/BiasAddGradBiasAddGradKdefault_policy/gradients/default_policy/model_2/lambda/Squeeze_grad/Reshape*
T0*
_output_shapes
:*
data_formatNHWC
?
\default_policy/gradients/default_policy/model_2/conv_value_out/BiasAdd_grad/tuple/group_depsNoOpX^default_policy/gradients/default_policy/model_2/conv_value_out/BiasAdd_grad/BiasAddGradL^default_policy/gradients/default_policy/model_2/lambda/Squeeze_grad/Reshape
?
ddefault_policy/gradients/default_policy/model_2/conv_value_out/BiasAdd_grad/tuple/control_dependencyIdentityKdefault_policy/gradients/default_policy/model_2/lambda/Squeeze_grad/Reshape]^default_policy/gradients/default_policy/model_2/conv_value_out/BiasAdd_grad/tuple/group_deps*
T0*^
_classT
RPloc:@default_policy/gradients/default_policy/model_2/lambda/Squeeze_grad/Reshape*/
_output_shapes
:?????????
?
fdefault_policy/gradients/default_policy/model_2/conv_value_out/BiasAdd_grad/tuple/control_dependency_1IdentityWdefault_policy/gradients/default_policy/model_2/conv_value_out/BiasAdd_grad/BiasAddGrad]^default_policy/gradients/default_policy/model_2/conv_value_out/BiasAdd_grad/tuple/group_deps*
T0*j
_class`
^\loc:@default_policy/gradients/default_policy/model_2/conv_value_out/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
?
<default_policy/gradients/default_policy/Squeeze_4_grad/ShapeShape'default_policy/model_2/conv_out/BiasAdd*
T0*
_output_shapes
:*
out_type0
?
>default_policy/gradients/default_policy/Squeeze_4_grad/ReshapeReshapeOdefault_policy/gradients/default_policy/truediv_2_grad/tuple/control_dependency<default_policy/gradients/default_policy/Squeeze_4_grad/Shape*
T0*
Tshape0*/
_output_shapes
:?????????
?
Qdefault_policy/gradients/default_policy/model_2/conv_value_out/Conv2D_grad/ShapeNShapeN(default_policy/model_2/conv_value_3/Relu;default_policy/model_2/conv_value_out/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::*
out_type0
?
^default_policy/gradients/default_policy/model_2/conv_value_out/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputQdefault_policy/gradients/default_policy/model_2/conv_value_out/Conv2D_grad/ShapeN;default_policy/model_2/conv_value_out/Conv2D/ReadVariableOpddefault_policy/gradients/default_policy/model_2/conv_value_out/BiasAdd_grad/tuple/control_dependency*
T0*0
_output_shapes
:??????????*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
_default_policy/gradients/default_policy/model_2/conv_value_out/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter(default_policy/model_2/conv_value_3/ReluSdefault_policy/gradients/default_policy/model_2/conv_value_out/Conv2D_grad/ShapeN:1ddefault_policy/gradients/default_policy/model_2/conv_value_out/BiasAdd_grad/tuple/control_dependency*
T0*'
_output_shapes
:?*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
[default_policy/gradients/default_policy/model_2/conv_value_out/Conv2D_grad/tuple/group_depsNoOp`^default_policy/gradients/default_policy/model_2/conv_value_out/Conv2D_grad/Conv2DBackpropFilter_^default_policy/gradients/default_policy/model_2/conv_value_out/Conv2D_grad/Conv2DBackpropInput
?
cdefault_policy/gradients/default_policy/model_2/conv_value_out/Conv2D_grad/tuple/control_dependencyIdentity^default_policy/gradients/default_policy/model_2/conv_value_out/Conv2D_grad/Conv2DBackpropInput\^default_policy/gradients/default_policy/model_2/conv_value_out/Conv2D_grad/tuple/group_deps*
T0*q
_classg
ecloc:@default_policy/gradients/default_policy/model_2/conv_value_out/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:??????????
?
edefault_policy/gradients/default_policy/model_2/conv_value_out/Conv2D_grad/tuple/control_dependency_1Identity_default_policy/gradients/default_policy/model_2/conv_value_out/Conv2D_grad/Conv2DBackpropFilter\^default_policy/gradients/default_policy/model_2/conv_value_out/Conv2D_grad/tuple/group_deps*
T0*r
_classh
fdloc:@default_policy/gradients/default_policy/model_2/conv_value_out/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:?
?
Qdefault_policy/gradients/default_policy/model_2/conv_out/BiasAdd_grad/BiasAddGradBiasAddGrad>default_policy/gradients/default_policy/Squeeze_4_grad/Reshape*
T0*
_output_shapes
:*
data_formatNHWC
?
Vdefault_policy/gradients/default_policy/model_2/conv_out/BiasAdd_grad/tuple/group_depsNoOp?^default_policy/gradients/default_policy/Squeeze_4_grad/ReshapeR^default_policy/gradients/default_policy/model_2/conv_out/BiasAdd_grad/BiasAddGrad
?
^default_policy/gradients/default_policy/model_2/conv_out/BiasAdd_grad/tuple/control_dependencyIdentity>default_policy/gradients/default_policy/Squeeze_4_grad/ReshapeW^default_policy/gradients/default_policy/model_2/conv_out/BiasAdd_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@default_policy/gradients/default_policy/Squeeze_4_grad/Reshape*/
_output_shapes
:?????????
?
`default_policy/gradients/default_policy/model_2/conv_out/BiasAdd_grad/tuple/control_dependency_1IdentityQdefault_policy/gradients/default_policy/model_2/conv_out/BiasAdd_grad/BiasAddGradW^default_policy/gradients/default_policy/model_2/conv_out/BiasAdd_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@default_policy/gradients/default_policy/model_2/conv_out/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
?
Odefault_policy/gradients/default_policy/model_2/conv_value_3/Relu_grad/ReluGradReluGradcdefault_policy/gradients/default_policy/model_2/conv_value_out/Conv2D_grad/tuple/control_dependency(default_policy/model_2/conv_value_3/Relu*
T0*0
_output_shapes
:??????????
?
Kdefault_policy/gradients/default_policy/model_2/conv_out/Conv2D_grad/ShapeNShapeN!default_policy/model_2/conv3/Relu5default_policy/model_2/conv_out/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::*
out_type0
?
Xdefault_policy/gradients/default_policy/model_2/conv_out/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputKdefault_policy/gradients/default_policy/model_2/conv_out/Conv2D_grad/ShapeN5default_policy/model_2/conv_out/Conv2D/ReadVariableOp^default_policy/gradients/default_policy/model_2/conv_out/BiasAdd_grad/tuple/control_dependency*
T0*0
_output_shapes
:??????????*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
Ydefault_policy/gradients/default_policy/model_2/conv_out/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter!default_policy/model_2/conv3/ReluMdefault_policy/gradients/default_policy/model_2/conv_out/Conv2D_grad/ShapeN:1^default_policy/gradients/default_policy/model_2/conv_out/BiasAdd_grad/tuple/control_dependency*
T0*'
_output_shapes
:?*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
Udefault_policy/gradients/default_policy/model_2/conv_out/Conv2D_grad/tuple/group_depsNoOpZ^default_policy/gradients/default_policy/model_2/conv_out/Conv2D_grad/Conv2DBackpropFilterY^default_policy/gradients/default_policy/model_2/conv_out/Conv2D_grad/Conv2DBackpropInput
?
]default_policy/gradients/default_policy/model_2/conv_out/Conv2D_grad/tuple/control_dependencyIdentityXdefault_policy/gradients/default_policy/model_2/conv_out/Conv2D_grad/Conv2DBackpropInputV^default_policy/gradients/default_policy/model_2/conv_out/Conv2D_grad/tuple/group_deps*
T0*k
_classa
_]loc:@default_policy/gradients/default_policy/model_2/conv_out/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:??????????
?
_default_policy/gradients/default_policy/model_2/conv_out/Conv2D_grad/tuple/control_dependency_1IdentityYdefault_policy/gradients/default_policy/model_2/conv_out/Conv2D_grad/Conv2DBackpropFilterV^default_policy/gradients/default_policy/model_2/conv_out/Conv2D_grad/tuple/group_deps*
T0*l
_classb
`^loc:@default_policy/gradients/default_policy/model_2/conv_out/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
:?
?
Udefault_policy/gradients/default_policy/model_2/conv_value_3/BiasAdd_grad/BiasAddGradBiasAddGradOdefault_policy/gradients/default_policy/model_2/conv_value_3/Relu_grad/ReluGrad*
T0*
_output_shapes	
:?*
data_formatNHWC
?
Zdefault_policy/gradients/default_policy/model_2/conv_value_3/BiasAdd_grad/tuple/group_depsNoOpV^default_policy/gradients/default_policy/model_2/conv_value_3/BiasAdd_grad/BiasAddGradP^default_policy/gradients/default_policy/model_2/conv_value_3/Relu_grad/ReluGrad
?
bdefault_policy/gradients/default_policy/model_2/conv_value_3/BiasAdd_grad/tuple/control_dependencyIdentityOdefault_policy/gradients/default_policy/model_2/conv_value_3/Relu_grad/ReluGrad[^default_policy/gradients/default_policy/model_2/conv_value_3/BiasAdd_grad/tuple/group_deps*
T0*b
_classX
VTloc:@default_policy/gradients/default_policy/model_2/conv_value_3/Relu_grad/ReluGrad*0
_output_shapes
:??????????
?
ddefault_policy/gradients/default_policy/model_2/conv_value_3/BiasAdd_grad/tuple/control_dependency_1IdentityUdefault_policy/gradients/default_policy/model_2/conv_value_3/BiasAdd_grad/BiasAddGrad[^default_policy/gradients/default_policy/model_2/conv_value_3/BiasAdd_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@default_policy/gradients/default_policy/model_2/conv_value_3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:?
?
Hdefault_policy/gradients/default_policy/model_2/conv3/Relu_grad/ReluGradReluGrad]default_policy/gradients/default_policy/model_2/conv_out/Conv2D_grad/tuple/control_dependency!default_policy/model_2/conv3/Relu*
T0*0
_output_shapes
:??????????
?
Odefault_policy/gradients/default_policy/model_2/conv_value_3/Conv2D_grad/ShapeNShapeN(default_policy/model_2/conv_value_2/Relu9default_policy/model_2/conv_value_3/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::*
out_type0
?
\default_policy/gradients/default_policy/model_2/conv_value_3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputOdefault_policy/gradients/default_policy/model_2/conv_value_3/Conv2D_grad/ShapeN9default_policy/model_2/conv_value_3/Conv2D/ReadVariableOpbdefault_policy/gradients/default_policy/model_2/conv_value_3/BiasAdd_grad/tuple/control_dependency*
T0*/
_output_shapes
:????????? *
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingVALID*
strides
*
use_cudnn_on_gpu(
?
]default_policy/gradients/default_policy/model_2/conv_value_3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter(default_policy/model_2/conv_value_2/ReluQdefault_policy/gradients/default_policy/model_2/conv_value_3/Conv2D_grad/ShapeN:1bdefault_policy/gradients/default_policy/model_2/conv_value_3/BiasAdd_grad/tuple/control_dependency*
T0*'
_output_shapes
: ?*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingVALID*
strides
*
use_cudnn_on_gpu(
?
Ydefault_policy/gradients/default_policy/model_2/conv_value_3/Conv2D_grad/tuple/group_depsNoOp^^default_policy/gradients/default_policy/model_2/conv_value_3/Conv2D_grad/Conv2DBackpropFilter]^default_policy/gradients/default_policy/model_2/conv_value_3/Conv2D_grad/Conv2DBackpropInput
?
adefault_policy/gradients/default_policy/model_2/conv_value_3/Conv2D_grad/tuple/control_dependencyIdentity\default_policy/gradients/default_policy/model_2/conv_value_3/Conv2D_grad/Conv2DBackpropInputZ^default_policy/gradients/default_policy/model_2/conv_value_3/Conv2D_grad/tuple/group_deps*
T0*o
_classe
caloc:@default_policy/gradients/default_policy/model_2/conv_value_3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:????????? 
?
cdefault_policy/gradients/default_policy/model_2/conv_value_3/Conv2D_grad/tuple/control_dependency_1Identity]default_policy/gradients/default_policy/model_2/conv_value_3/Conv2D_grad/Conv2DBackpropFilterZ^default_policy/gradients/default_policy/model_2/conv_value_3/Conv2D_grad/tuple/group_deps*
T0*p
_classf
dbloc:@default_policy/gradients/default_policy/model_2/conv_value_3/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
: ?
?
Ndefault_policy/gradients/default_policy/model_2/conv3/BiasAdd_grad/BiasAddGradBiasAddGradHdefault_policy/gradients/default_policy/model_2/conv3/Relu_grad/ReluGrad*
T0*
_output_shapes	
:?*
data_formatNHWC
?
Sdefault_policy/gradients/default_policy/model_2/conv3/BiasAdd_grad/tuple/group_depsNoOpO^default_policy/gradients/default_policy/model_2/conv3/BiasAdd_grad/BiasAddGradI^default_policy/gradients/default_policy/model_2/conv3/Relu_grad/ReluGrad
?
[default_policy/gradients/default_policy/model_2/conv3/BiasAdd_grad/tuple/control_dependencyIdentityHdefault_policy/gradients/default_policy/model_2/conv3/Relu_grad/ReluGradT^default_policy/gradients/default_policy/model_2/conv3/BiasAdd_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@default_policy/gradients/default_policy/model_2/conv3/Relu_grad/ReluGrad*0
_output_shapes
:??????????
?
]default_policy/gradients/default_policy/model_2/conv3/BiasAdd_grad/tuple/control_dependency_1IdentityNdefault_policy/gradients/default_policy/model_2/conv3/BiasAdd_grad/BiasAddGradT^default_policy/gradients/default_policy/model_2/conv3/BiasAdd_grad/tuple/group_deps*
T0*a
_classW
USloc:@default_policy/gradients/default_policy/model_2/conv3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:?
?
Odefault_policy/gradients/default_policy/model_2/conv_value_2/Relu_grad/ReluGradReluGradadefault_policy/gradients/default_policy/model_2/conv_value_3/Conv2D_grad/tuple/control_dependency(default_policy/model_2/conv_value_2/Relu*
T0*/
_output_shapes
:????????? 
?
Hdefault_policy/gradients/default_policy/model_2/conv3/Conv2D_grad/ShapeNShapeN!default_policy/model_2/conv2/Relu2default_policy/model_2/conv3/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::*
out_type0
?
Udefault_policy/gradients/default_policy/model_2/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputHdefault_policy/gradients/default_policy/model_2/conv3/Conv2D_grad/ShapeN2default_policy/model_2/conv3/Conv2D/ReadVariableOp[default_policy/gradients/default_policy/model_2/conv3/BiasAdd_grad/tuple/control_dependency*
T0*/
_output_shapes
:????????? *
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingVALID*
strides
*
use_cudnn_on_gpu(
?
Vdefault_policy/gradients/default_policy/model_2/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter!default_policy/model_2/conv2/ReluJdefault_policy/gradients/default_policy/model_2/conv3/Conv2D_grad/ShapeN:1[default_policy/gradients/default_policy/model_2/conv3/BiasAdd_grad/tuple/control_dependency*
T0*'
_output_shapes
: ?*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingVALID*
strides
*
use_cudnn_on_gpu(
?
Rdefault_policy/gradients/default_policy/model_2/conv3/Conv2D_grad/tuple/group_depsNoOpW^default_policy/gradients/default_policy/model_2/conv3/Conv2D_grad/Conv2DBackpropFilterV^default_policy/gradients/default_policy/model_2/conv3/Conv2D_grad/Conv2DBackpropInput
?
Zdefault_policy/gradients/default_policy/model_2/conv3/Conv2D_grad/tuple/control_dependencyIdentityUdefault_policy/gradients/default_policy/model_2/conv3/Conv2D_grad/Conv2DBackpropInputS^default_policy/gradients/default_policy/model_2/conv3/Conv2D_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@default_policy/gradients/default_policy/model_2/conv3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:????????? 
?
\default_policy/gradients/default_policy/model_2/conv3/Conv2D_grad/tuple/control_dependency_1IdentityVdefault_policy/gradients/default_policy/model_2/conv3/Conv2D_grad/Conv2DBackpropFilterS^default_policy/gradients/default_policy/model_2/conv3/Conv2D_grad/tuple/group_deps*
T0*i
_class_
][loc:@default_policy/gradients/default_policy/model_2/conv3/Conv2D_grad/Conv2DBackpropFilter*'
_output_shapes
: ?
?
Udefault_policy/gradients/default_policy/model_2/conv_value_2/BiasAdd_grad/BiasAddGradBiasAddGradOdefault_policy/gradients/default_policy/model_2/conv_value_2/Relu_grad/ReluGrad*
T0*
_output_shapes
: *
data_formatNHWC
?
Zdefault_policy/gradients/default_policy/model_2/conv_value_2/BiasAdd_grad/tuple/group_depsNoOpV^default_policy/gradients/default_policy/model_2/conv_value_2/BiasAdd_grad/BiasAddGradP^default_policy/gradients/default_policy/model_2/conv_value_2/Relu_grad/ReluGrad
?
bdefault_policy/gradients/default_policy/model_2/conv_value_2/BiasAdd_grad/tuple/control_dependencyIdentityOdefault_policy/gradients/default_policy/model_2/conv_value_2/Relu_grad/ReluGrad[^default_policy/gradients/default_policy/model_2/conv_value_2/BiasAdd_grad/tuple/group_deps*
T0*b
_classX
VTloc:@default_policy/gradients/default_policy/model_2/conv_value_2/Relu_grad/ReluGrad*/
_output_shapes
:????????? 
?
ddefault_policy/gradients/default_policy/model_2/conv_value_2/BiasAdd_grad/tuple/control_dependency_1IdentityUdefault_policy/gradients/default_policy/model_2/conv_value_2/BiasAdd_grad/BiasAddGrad[^default_policy/gradients/default_policy/model_2/conv_value_2/BiasAdd_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@default_policy/gradients/default_policy/model_2/conv_value_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
?
Hdefault_policy/gradients/default_policy/model_2/conv2/Relu_grad/ReluGradReluGradZdefault_policy/gradients/default_policy/model_2/conv3/Conv2D_grad/tuple/control_dependency!default_policy/model_2/conv2/Relu*
T0*/
_output_shapes
:????????? 
?
Odefault_policy/gradients/default_policy/model_2/conv_value_2/Conv2D_grad/ShapeNShapeN(default_policy/model_2/conv_value_1/Relu9default_policy/model_2/conv_value_2/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::*
out_type0
?
\default_policy/gradients/default_policy/model_2/conv_value_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputOdefault_policy/gradients/default_policy/model_2/conv_value_2/Conv2D_grad/ShapeN9default_policy/model_2/conv_value_2/Conv2D/ReadVariableOpbdefault_policy/gradients/default_policy/model_2/conv_value_2/BiasAdd_grad/tuple/control_dependency*
T0*/
_output_shapes
:?????????*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
]default_policy/gradients/default_policy/model_2/conv_value_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter(default_policy/model_2/conv_value_1/ReluQdefault_policy/gradients/default_policy/model_2/conv_value_2/Conv2D_grad/ShapeN:1bdefault_policy/gradients/default_policy/model_2/conv_value_2/BiasAdd_grad/tuple/control_dependency*
T0*&
_output_shapes
: *
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
Ydefault_policy/gradients/default_policy/model_2/conv_value_2/Conv2D_grad/tuple/group_depsNoOp^^default_policy/gradients/default_policy/model_2/conv_value_2/Conv2D_grad/Conv2DBackpropFilter]^default_policy/gradients/default_policy/model_2/conv_value_2/Conv2D_grad/Conv2DBackpropInput
?
adefault_policy/gradients/default_policy/model_2/conv_value_2/Conv2D_grad/tuple/control_dependencyIdentity\default_policy/gradients/default_policy/model_2/conv_value_2/Conv2D_grad/Conv2DBackpropInputZ^default_policy/gradients/default_policy/model_2/conv_value_2/Conv2D_grad/tuple/group_deps*
T0*o
_classe
caloc:@default_policy/gradients/default_policy/model_2/conv_value_2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:?????????
?
cdefault_policy/gradients/default_policy/model_2/conv_value_2/Conv2D_grad/tuple/control_dependency_1Identity]default_policy/gradients/default_policy/model_2/conv_value_2/Conv2D_grad/Conv2DBackpropFilterZ^default_policy/gradients/default_policy/model_2/conv_value_2/Conv2D_grad/tuple/group_deps*
T0*p
_classf
dbloc:@default_policy/gradients/default_policy/model_2/conv_value_2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
?
Ndefault_policy/gradients/default_policy/model_2/conv2/BiasAdd_grad/BiasAddGradBiasAddGradHdefault_policy/gradients/default_policy/model_2/conv2/Relu_grad/ReluGrad*
T0*
_output_shapes
: *
data_formatNHWC
?
Sdefault_policy/gradients/default_policy/model_2/conv2/BiasAdd_grad/tuple/group_depsNoOpO^default_policy/gradients/default_policy/model_2/conv2/BiasAdd_grad/BiasAddGradI^default_policy/gradients/default_policy/model_2/conv2/Relu_grad/ReluGrad
?
[default_policy/gradients/default_policy/model_2/conv2/BiasAdd_grad/tuple/control_dependencyIdentityHdefault_policy/gradients/default_policy/model_2/conv2/Relu_grad/ReluGradT^default_policy/gradients/default_policy/model_2/conv2/BiasAdd_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@default_policy/gradients/default_policy/model_2/conv2/Relu_grad/ReluGrad*/
_output_shapes
:????????? 
?
]default_policy/gradients/default_policy/model_2/conv2/BiasAdd_grad/tuple/control_dependency_1IdentityNdefault_policy/gradients/default_policy/model_2/conv2/BiasAdd_grad/BiasAddGradT^default_policy/gradients/default_policy/model_2/conv2/BiasAdd_grad/tuple/group_deps*
T0*a
_classW
USloc:@default_policy/gradients/default_policy/model_2/conv2/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
?
Odefault_policy/gradients/default_policy/model_2/conv_value_1/Relu_grad/ReluGradReluGradadefault_policy/gradients/default_policy/model_2/conv_value_2/Conv2D_grad/tuple/control_dependency(default_policy/model_2/conv_value_1/Relu*
T0*/
_output_shapes
:?????????
?
Hdefault_policy/gradients/default_policy/model_2/conv2/Conv2D_grad/ShapeNShapeN!default_policy/model_2/conv1/Relu2default_policy/model_2/conv2/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::*
out_type0
?
Udefault_policy/gradients/default_policy/model_2/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputHdefault_policy/gradients/default_policy/model_2/conv2/Conv2D_grad/ShapeN2default_policy/model_2/conv2/Conv2D/ReadVariableOp[default_policy/gradients/default_policy/model_2/conv2/BiasAdd_grad/tuple/control_dependency*
T0*/
_output_shapes
:?????????*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
Vdefault_policy/gradients/default_policy/model_2/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter!default_policy/model_2/conv1/ReluJdefault_policy/gradients/default_policy/model_2/conv2/Conv2D_grad/ShapeN:1[default_policy/gradients/default_policy/model_2/conv2/BiasAdd_grad/tuple/control_dependency*
T0*&
_output_shapes
: *
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
Rdefault_policy/gradients/default_policy/model_2/conv2/Conv2D_grad/tuple/group_depsNoOpW^default_policy/gradients/default_policy/model_2/conv2/Conv2D_grad/Conv2DBackpropFilterV^default_policy/gradients/default_policy/model_2/conv2/Conv2D_grad/Conv2DBackpropInput
?
Zdefault_policy/gradients/default_policy/model_2/conv2/Conv2D_grad/tuple/control_dependencyIdentityUdefault_policy/gradients/default_policy/model_2/conv2/Conv2D_grad/Conv2DBackpropInputS^default_policy/gradients/default_policy/model_2/conv2/Conv2D_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@default_policy/gradients/default_policy/model_2/conv2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:?????????
?
\default_policy/gradients/default_policy/model_2/conv2/Conv2D_grad/tuple/control_dependency_1IdentityVdefault_policy/gradients/default_policy/model_2/conv2/Conv2D_grad/Conv2DBackpropFilterS^default_policy/gradients/default_policy/model_2/conv2/Conv2D_grad/tuple/group_deps*
T0*i
_class_
][loc:@default_policy/gradients/default_policy/model_2/conv2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
?
Udefault_policy/gradients/default_policy/model_2/conv_value_1/BiasAdd_grad/BiasAddGradBiasAddGradOdefault_policy/gradients/default_policy/model_2/conv_value_1/Relu_grad/ReluGrad*
T0*
_output_shapes
:*
data_formatNHWC
?
Zdefault_policy/gradients/default_policy/model_2/conv_value_1/BiasAdd_grad/tuple/group_depsNoOpV^default_policy/gradients/default_policy/model_2/conv_value_1/BiasAdd_grad/BiasAddGradP^default_policy/gradients/default_policy/model_2/conv_value_1/Relu_grad/ReluGrad
?
bdefault_policy/gradients/default_policy/model_2/conv_value_1/BiasAdd_grad/tuple/control_dependencyIdentityOdefault_policy/gradients/default_policy/model_2/conv_value_1/Relu_grad/ReluGrad[^default_policy/gradients/default_policy/model_2/conv_value_1/BiasAdd_grad/tuple/group_deps*
T0*b
_classX
VTloc:@default_policy/gradients/default_policy/model_2/conv_value_1/Relu_grad/ReluGrad*/
_output_shapes
:?????????
?
ddefault_policy/gradients/default_policy/model_2/conv_value_1/BiasAdd_grad/tuple/control_dependency_1IdentityUdefault_policy/gradients/default_policy/model_2/conv_value_1/BiasAdd_grad/BiasAddGrad[^default_policy/gradients/default_policy/model_2/conv_value_1/BiasAdd_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@default_policy/gradients/default_policy/model_2/conv_value_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
?
Hdefault_policy/gradients/default_policy/model_2/conv1/Relu_grad/ReluGradReluGradZdefault_policy/gradients/default_policy/model_2/conv2/Conv2D_grad/tuple/control_dependency!default_policy/model_2/conv1/Relu*
T0*/
_output_shapes
:?????????
?
Odefault_policy/gradients/default_policy/model_2/conv_value_1/Conv2D_grad/ShapeNShapeNdefault_policy/Cast_59default_policy/model_2/conv_value_1/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::*
out_type0
?
\default_policy/gradients/default_policy/model_2/conv_value_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputOdefault_policy/gradients/default_policy/model_2/conv_value_1/Conv2D_grad/ShapeN9default_policy/model_2/conv_value_1/Conv2D/ReadVariableOpbdefault_policy/gradients/default_policy/model_2/conv_value_1/BiasAdd_grad/tuple/control_dependency*
T0*/
_output_shapes
:?????????TT*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
]default_policy/gradients/default_policy/model_2/conv_value_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterdefault_policy/Cast_5Qdefault_policy/gradients/default_policy/model_2/conv_value_1/Conv2D_grad/ShapeN:1bdefault_policy/gradients/default_policy/model_2/conv_value_1/BiasAdd_grad/tuple/control_dependency*
T0*&
_output_shapes
:*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
Ydefault_policy/gradients/default_policy/model_2/conv_value_1/Conv2D_grad/tuple/group_depsNoOp^^default_policy/gradients/default_policy/model_2/conv_value_1/Conv2D_grad/Conv2DBackpropFilter]^default_policy/gradients/default_policy/model_2/conv_value_1/Conv2D_grad/Conv2DBackpropInput
?
adefault_policy/gradients/default_policy/model_2/conv_value_1/Conv2D_grad/tuple/control_dependencyIdentity\default_policy/gradients/default_policy/model_2/conv_value_1/Conv2D_grad/Conv2DBackpropInputZ^default_policy/gradients/default_policy/model_2/conv_value_1/Conv2D_grad/tuple/group_deps*
T0*o
_classe
caloc:@default_policy/gradients/default_policy/model_2/conv_value_1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:?????????TT
?
cdefault_policy/gradients/default_policy/model_2/conv_value_1/Conv2D_grad/tuple/control_dependency_1Identity]default_policy/gradients/default_policy/model_2/conv_value_1/Conv2D_grad/Conv2DBackpropFilterZ^default_policy/gradients/default_policy/model_2/conv_value_1/Conv2D_grad/tuple/group_deps*
T0*p
_classf
dbloc:@default_policy/gradients/default_policy/model_2/conv_value_1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
?
Ndefault_policy/gradients/default_policy/model_2/conv1/BiasAdd_grad/BiasAddGradBiasAddGradHdefault_policy/gradients/default_policy/model_2/conv1/Relu_grad/ReluGrad*
T0*
_output_shapes
:*
data_formatNHWC
?
Sdefault_policy/gradients/default_policy/model_2/conv1/BiasAdd_grad/tuple/group_depsNoOpO^default_policy/gradients/default_policy/model_2/conv1/BiasAdd_grad/BiasAddGradI^default_policy/gradients/default_policy/model_2/conv1/Relu_grad/ReluGrad
?
[default_policy/gradients/default_policy/model_2/conv1/BiasAdd_grad/tuple/control_dependencyIdentityHdefault_policy/gradients/default_policy/model_2/conv1/Relu_grad/ReluGradT^default_policy/gradients/default_policy/model_2/conv1/BiasAdd_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@default_policy/gradients/default_policy/model_2/conv1/Relu_grad/ReluGrad*/
_output_shapes
:?????????
?
]default_policy/gradients/default_policy/model_2/conv1/BiasAdd_grad/tuple/control_dependency_1IdentityNdefault_policy/gradients/default_policy/model_2/conv1/BiasAdd_grad/BiasAddGradT^default_policy/gradients/default_policy/model_2/conv1/BiasAdd_grad/tuple/group_deps*
T0*a
_classW
USloc:@default_policy/gradients/default_policy/model_2/conv1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
?
Hdefault_policy/gradients/default_policy/model_2/conv1/Conv2D_grad/ShapeNShapeNdefault_policy/Cast_52default_policy/model_2/conv1/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::*
out_type0
?
Udefault_policy/gradients/default_policy/model_2/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputHdefault_policy/gradients/default_policy/model_2/conv1/Conv2D_grad/ShapeN2default_policy/model_2/conv1/Conv2D/ReadVariableOp[default_policy/gradients/default_policy/model_2/conv1/BiasAdd_grad/tuple/control_dependency*
T0*/
_output_shapes
:?????????TT*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
Vdefault_policy/gradients/default_policy/model_2/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterdefault_policy/Cast_5Jdefault_policy/gradients/default_policy/model_2/conv1/Conv2D_grad/ShapeN:1[default_policy/gradients/default_policy/model_2/conv1/BiasAdd_grad/tuple/control_dependency*
T0*&
_output_shapes
:*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
?
Rdefault_policy/gradients/default_policy/model_2/conv1/Conv2D_grad/tuple/group_depsNoOpW^default_policy/gradients/default_policy/model_2/conv1/Conv2D_grad/Conv2DBackpropFilterV^default_policy/gradients/default_policy/model_2/conv1/Conv2D_grad/Conv2DBackpropInput
?
Zdefault_policy/gradients/default_policy/model_2/conv1/Conv2D_grad/tuple/control_dependencyIdentityUdefault_policy/gradients/default_policy/model_2/conv1/Conv2D_grad/Conv2DBackpropInputS^default_policy/gradients/default_policy/model_2/conv1/Conv2D_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@default_policy/gradients/default_policy/model_2/conv1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:?????????TT
?
\default_policy/gradients/default_policy/model_2/conv1/Conv2D_grad/tuple/control_dependency_1IdentityVdefault_policy/gradients/default_policy/model_2/conv1/Conv2D_grad/Conv2DBackpropFilterS^default_policy/gradients/default_policy/model_2/conv1/Conv2D_grad/tuple/group_deps*
T0*i
_class_
][loc:@default_policy/gradients/default_policy/model_2/conv1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
?
default_policy/ReadVariableOp_4ReadVariableOp"default_policy/conv_value_1/kernel*&
_output_shapes
:*
dtype0
?
=default_policy/Placeholder_default_policy/conv_value_1/kernelPlaceholder*&
_output_shapes
:*
dtype0*
shape:
?
!default_policy/AssignVariableOp_2AssignVariableOp"default_policy/conv_value_1/kernel=default_policy/Placeholder_default_policy/conv_value_1/kernel*
dtype0
?
default_policy/ReadVariableOp_5ReadVariableOp"default_policy/conv_value_1/kernel"^default_policy/AssignVariableOp_2*&
_output_shapes
:*
dtype0
|
default_policy/ReadVariableOp_6ReadVariableOp default_policy/conv_value_1/bias*
_output_shapes
:*
dtype0
?
;default_policy/Placeholder_default_policy/conv_value_1/biasPlaceholder*
_output_shapes
:*
dtype0*
shape:
?
!default_policy/AssignVariableOp_3AssignVariableOp default_policy/conv_value_1/bias;default_policy/Placeholder_default_policy/conv_value_1/bias*
dtype0
?
default_policy/ReadVariableOp_7ReadVariableOp default_policy/conv_value_1/bias"^default_policy/AssignVariableOp_3*
_output_shapes
:*
dtype0
?
default_policy/ReadVariableOp_8ReadVariableOpdefault_policy/conv1/kernel*&
_output_shapes
:*
dtype0
?
6default_policy/Placeholder_default_policy/conv1/kernelPlaceholder*&
_output_shapes
:*
dtype0*
shape:
?
!default_policy/AssignVariableOp_4AssignVariableOpdefault_policy/conv1/kernel6default_policy/Placeholder_default_policy/conv1/kernel*
dtype0
?
default_policy/ReadVariableOp_9ReadVariableOpdefault_policy/conv1/kernel"^default_policy/AssignVariableOp_4*&
_output_shapes
:*
dtype0
v
 default_policy/ReadVariableOp_10ReadVariableOpdefault_policy/conv1/bias*
_output_shapes
:*
dtype0
}
4default_policy/Placeholder_default_policy/conv1/biasPlaceholder*
_output_shapes
:*
dtype0*
shape:
?
!default_policy/AssignVariableOp_5AssignVariableOpdefault_policy/conv1/bias4default_policy/Placeholder_default_policy/conv1/bias*
dtype0
?
 default_policy/ReadVariableOp_11ReadVariableOpdefault_policy/conv1/bias"^default_policy/AssignVariableOp_5*
_output_shapes
:*
dtype0
?
 default_policy/ReadVariableOp_12ReadVariableOp"default_policy/conv_value_2/kernel*&
_output_shapes
: *
dtype0
?
=default_policy/Placeholder_default_policy/conv_value_2/kernelPlaceholder*&
_output_shapes
: *
dtype0*
shape: 
?
!default_policy/AssignVariableOp_6AssignVariableOp"default_policy/conv_value_2/kernel=default_policy/Placeholder_default_policy/conv_value_2/kernel*
dtype0
?
 default_policy/ReadVariableOp_13ReadVariableOp"default_policy/conv_value_2/kernel"^default_policy/AssignVariableOp_6*&
_output_shapes
: *
dtype0
}
 default_policy/ReadVariableOp_14ReadVariableOp default_policy/conv_value_2/bias*
_output_shapes
: *
dtype0
?
;default_policy/Placeholder_default_policy/conv_value_2/biasPlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
!default_policy/AssignVariableOp_7AssignVariableOp default_policy/conv_value_2/bias;default_policy/Placeholder_default_policy/conv_value_2/bias*
dtype0
?
 default_policy/ReadVariableOp_15ReadVariableOp default_policy/conv_value_2/bias"^default_policy/AssignVariableOp_7*
_output_shapes
: *
dtype0
?
 default_policy/ReadVariableOp_16ReadVariableOpdefault_policy/conv2/kernel*&
_output_shapes
: *
dtype0
?
6default_policy/Placeholder_default_policy/conv2/kernelPlaceholder*&
_output_shapes
: *
dtype0*
shape: 
?
!default_policy/AssignVariableOp_8AssignVariableOpdefault_policy/conv2/kernel6default_policy/Placeholder_default_policy/conv2/kernel*
dtype0
?
 default_policy/ReadVariableOp_17ReadVariableOpdefault_policy/conv2/kernel"^default_policy/AssignVariableOp_8*&
_output_shapes
: *
dtype0
v
 default_policy/ReadVariableOp_18ReadVariableOpdefault_policy/conv2/bias*
_output_shapes
: *
dtype0
}
4default_policy/Placeholder_default_policy/conv2/biasPlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
!default_policy/AssignVariableOp_9AssignVariableOpdefault_policy/conv2/bias4default_policy/Placeholder_default_policy/conv2/bias*
dtype0
?
 default_policy/ReadVariableOp_19ReadVariableOpdefault_policy/conv2/bias"^default_policy/AssignVariableOp_9*
_output_shapes
: *
dtype0
?
 default_policy/ReadVariableOp_20ReadVariableOp"default_policy/conv_value_3/kernel*'
_output_shapes
: ?*
dtype0
?
=default_policy/Placeholder_default_policy/conv_value_3/kernelPlaceholder*'
_output_shapes
: ?*
dtype0*
shape: ?
?
"default_policy/AssignVariableOp_10AssignVariableOp"default_policy/conv_value_3/kernel=default_policy/Placeholder_default_policy/conv_value_3/kernel*
dtype0
?
 default_policy/ReadVariableOp_21ReadVariableOp"default_policy/conv_value_3/kernel#^default_policy/AssignVariableOp_10*'
_output_shapes
: ?*
dtype0
~
 default_policy/ReadVariableOp_22ReadVariableOp default_policy/conv_value_3/bias*
_output_shapes	
:?*
dtype0
?
;default_policy/Placeholder_default_policy/conv_value_3/biasPlaceholder*
_output_shapes	
:?*
dtype0*
shape:?
?
"default_policy/AssignVariableOp_11AssignVariableOp default_policy/conv_value_3/bias;default_policy/Placeholder_default_policy/conv_value_3/bias*
dtype0
?
 default_policy/ReadVariableOp_23ReadVariableOp default_policy/conv_value_3/bias#^default_policy/AssignVariableOp_11*
_output_shapes	
:?*
dtype0
?
 default_policy/ReadVariableOp_24ReadVariableOpdefault_policy/conv3/kernel*'
_output_shapes
: ?*
dtype0
?
6default_policy/Placeholder_default_policy/conv3/kernelPlaceholder*'
_output_shapes
: ?*
dtype0*
shape: ?
?
"default_policy/AssignVariableOp_12AssignVariableOpdefault_policy/conv3/kernel6default_policy/Placeholder_default_policy/conv3/kernel*
dtype0
?
 default_policy/ReadVariableOp_25ReadVariableOpdefault_policy/conv3/kernel#^default_policy/AssignVariableOp_12*'
_output_shapes
: ?*
dtype0
w
 default_policy/ReadVariableOp_26ReadVariableOpdefault_policy/conv3/bias*
_output_shapes	
:?*
dtype0

4default_policy/Placeholder_default_policy/conv3/biasPlaceholder*
_output_shapes	
:?*
dtype0*
shape:?
?
"default_policy/AssignVariableOp_13AssignVariableOpdefault_policy/conv3/bias4default_policy/Placeholder_default_policy/conv3/bias*
dtype0
?
 default_policy/ReadVariableOp_27ReadVariableOpdefault_policy/conv3/bias#^default_policy/AssignVariableOp_13*
_output_shapes	
:?*
dtype0
?
 default_policy/ReadVariableOp_28ReadVariableOp$default_policy/conv_value_out/kernel*'
_output_shapes
:?*
dtype0
?
?default_policy/Placeholder_default_policy/conv_value_out/kernelPlaceholder*'
_output_shapes
:?*
dtype0*
shape:?
?
"default_policy/AssignVariableOp_14AssignVariableOp$default_policy/conv_value_out/kernel?default_policy/Placeholder_default_policy/conv_value_out/kernel*
dtype0
?
 default_policy/ReadVariableOp_29ReadVariableOp$default_policy/conv_value_out/kernel#^default_policy/AssignVariableOp_14*'
_output_shapes
:?*
dtype0

 default_policy/ReadVariableOp_30ReadVariableOp"default_policy/conv_value_out/bias*
_output_shapes
:*
dtype0
?
=default_policy/Placeholder_default_policy/conv_value_out/biasPlaceholder*
_output_shapes
:*
dtype0*
shape:
?
"default_policy/AssignVariableOp_15AssignVariableOp"default_policy/conv_value_out/bias=default_policy/Placeholder_default_policy/conv_value_out/bias*
dtype0
?
 default_policy/ReadVariableOp_31ReadVariableOp"default_policy/conv_value_out/bias#^default_policy/AssignVariableOp_15*
_output_shapes
:*
dtype0
?
 default_policy/ReadVariableOp_32ReadVariableOpdefault_policy/conv_out/kernel*'
_output_shapes
:?*
dtype0
?
9default_policy/Placeholder_default_policy/conv_out/kernelPlaceholder*'
_output_shapes
:?*
dtype0*
shape:?
?
"default_policy/AssignVariableOp_16AssignVariableOpdefault_policy/conv_out/kernel9default_policy/Placeholder_default_policy/conv_out/kernel*
dtype0
?
 default_policy/ReadVariableOp_33ReadVariableOpdefault_policy/conv_out/kernel#^default_policy/AssignVariableOp_16*'
_output_shapes
:?*
dtype0
y
 default_policy/ReadVariableOp_34ReadVariableOpdefault_policy/conv_out/bias*
_output_shapes
:*
dtype0
?
7default_policy/Placeholder_default_policy/conv_out/biasPlaceholder*
_output_shapes
:*
dtype0*
shape:
?
"default_policy/AssignVariableOp_17AssignVariableOpdefault_policy/conv_out/bias7default_policy/Placeholder_default_policy/conv_out/bias*
dtype0
?
 default_policy/ReadVariableOp_35ReadVariableOpdefault_policy/conv_out/bias#^default_policy/AssignVariableOp_17*
_output_shapes
:*
dtype0
?
,default_policy/global_step/Initializer/zerosConst*-
_class#
!loc:@default_policy/global_step*
_output_shapes
: *
dtype0	*
value	B	 R 
?
default_policy/global_stepVarHandleOp*-
_class#
!loc:@default_policy/global_step*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0	*
shape: *+
shared_namedefault_policy/global_step
?
;default_policy/global_step/IsInitialized/VarIsInitializedOpVarIsInitializedOpdefault_policy/global_step*
_output_shapes
: 
?
!default_policy/global_step/AssignAssignVariableOpdefault_policy/global_step,default_policy/global_step/Initializer/zeros*
dtype0	
?
.default_policy/global_step/Read/ReadVariableOpReadVariableOpdefault_policy/global_step*
_output_shapes
: *
dtype0	
?
4default_policy/beta1_power/Initializer/initial_valueConst*,
_class"
 loc:@default_policy/conv1/bias*
_output_shapes
: *
dtype0*
valueB
 *fff?
?
default_policy/beta1_powerVarHandleOp*,
_class"
 loc:@default_policy/conv1/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *+
shared_namedefault_policy/beta1_power
?
;default_policy/beta1_power/IsInitialized/VarIsInitializedOpVarIsInitializedOpdefault_policy/beta1_power*,
_class"
 loc:@default_policy/conv1/bias*
_output_shapes
: 
?
!default_policy/beta1_power/AssignAssignVariableOpdefault_policy/beta1_power4default_policy/beta1_power/Initializer/initial_value*
dtype0
?
.default_policy/beta1_power/Read/ReadVariableOpReadVariableOpdefault_policy/beta1_power*,
_class"
 loc:@default_policy/conv1/bias*
_output_shapes
: *
dtype0
?
4default_policy/beta2_power/Initializer/initial_valueConst*,
_class"
 loc:@default_policy/conv1/bias*
_output_shapes
: *
dtype0*
valueB
 *w??
?
default_policy/beta2_powerVarHandleOp*,
_class"
 loc:@default_policy/conv1/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *+
shared_namedefault_policy/beta2_power
?
;default_policy/beta2_power/IsInitialized/VarIsInitializedOpVarIsInitializedOpdefault_policy/beta2_power*,
_class"
 loc:@default_policy/conv1/bias*
_output_shapes
: 
?
!default_policy/beta2_power/AssignAssignVariableOpdefault_policy/beta2_power4default_policy/beta2_power/Initializer/initial_value*
dtype0
?
.default_policy/beta2_power/Read/ReadVariableOpReadVariableOpdefault_policy/beta2_power*,
_class"
 loc:@default_policy/conv1/bias*
_output_shapes
: *
dtype0
?
Xdefault_policy/default_policy/conv_value_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*5
_class+
)'loc:@default_policy/conv_value_1/kernel*
_output_shapes
:*
dtype0*%
valueB"            
?
Ndefault_policy/default_policy/conv_value_1/kernel/Adam/Initializer/zeros/ConstConst*5
_class+
)'loc:@default_policy/conv_value_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
?
Hdefault_policy/default_policy/conv_value_1/kernel/Adam/Initializer/zerosFillXdefault_policy/default_policy/conv_value_1/kernel/Adam/Initializer/zeros/shape_as_tensorNdefault_policy/default_policy/conv_value_1/kernel/Adam/Initializer/zeros/Const*
T0*5
_class+
)'loc:@default_policy/conv_value_1/kernel*&
_output_shapes
:*

index_type0
?
6default_policy/default_policy/conv_value_1/kernel/AdamVarHandleOp*5
_class+
)'loc:@default_policy/conv_value_1/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*G
shared_name86default_policy/default_policy/conv_value_1/kernel/Adam
?
Wdefault_policy/default_policy/conv_value_1/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp6default_policy/default_policy/conv_value_1/kernel/Adam*5
_class+
)'loc:@default_policy/conv_value_1/kernel*
_output_shapes
: 
?
=default_policy/default_policy/conv_value_1/kernel/Adam/AssignAssignVariableOp6default_policy/default_policy/conv_value_1/kernel/AdamHdefault_policy/default_policy/conv_value_1/kernel/Adam/Initializer/zeros*
dtype0
?
Jdefault_policy/default_policy/conv_value_1/kernel/Adam/Read/ReadVariableOpReadVariableOp6default_policy/default_policy/conv_value_1/kernel/Adam*5
_class+
)'loc:@default_policy/conv_value_1/kernel*&
_output_shapes
:*
dtype0
?
Zdefault_policy/default_policy/conv_value_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*5
_class+
)'loc:@default_policy/conv_value_1/kernel*
_output_shapes
:*
dtype0*%
valueB"            
?
Pdefault_policy/default_policy/conv_value_1/kernel/Adam_1/Initializer/zeros/ConstConst*5
_class+
)'loc:@default_policy/conv_value_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
?
Jdefault_policy/default_policy/conv_value_1/kernel/Adam_1/Initializer/zerosFillZdefault_policy/default_policy/conv_value_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorPdefault_policy/default_policy/conv_value_1/kernel/Adam_1/Initializer/zeros/Const*
T0*5
_class+
)'loc:@default_policy/conv_value_1/kernel*&
_output_shapes
:*

index_type0
?
8default_policy/default_policy/conv_value_1/kernel/Adam_1VarHandleOp*5
_class+
)'loc:@default_policy/conv_value_1/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*I
shared_name:8default_policy/default_policy/conv_value_1/kernel/Adam_1
?
Ydefault_policy/default_policy/conv_value_1/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp8default_policy/default_policy/conv_value_1/kernel/Adam_1*5
_class+
)'loc:@default_policy/conv_value_1/kernel*
_output_shapes
: 
?
?default_policy/default_policy/conv_value_1/kernel/Adam_1/AssignAssignVariableOp8default_policy/default_policy/conv_value_1/kernel/Adam_1Jdefault_policy/default_policy/conv_value_1/kernel/Adam_1/Initializer/zeros*
dtype0
?
Ldefault_policy/default_policy/conv_value_1/kernel/Adam_1/Read/ReadVariableOpReadVariableOp8default_policy/default_policy/conv_value_1/kernel/Adam_1*5
_class+
)'loc:@default_policy/conv_value_1/kernel*&
_output_shapes
:*
dtype0
?
Fdefault_policy/default_policy/conv_value_1/bias/Adam/Initializer/zerosConst*3
_class)
'%loc:@default_policy/conv_value_1/bias*
_output_shapes
:*
dtype0*
valueB*    
?
4default_policy/default_policy/conv_value_1/bias/AdamVarHandleOp*3
_class)
'%loc:@default_policy/conv_value_1/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*E
shared_name64default_policy/default_policy/conv_value_1/bias/Adam
?
Udefault_policy/default_policy/conv_value_1/bias/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp4default_policy/default_policy/conv_value_1/bias/Adam*3
_class)
'%loc:@default_policy/conv_value_1/bias*
_output_shapes
: 
?
;default_policy/default_policy/conv_value_1/bias/Adam/AssignAssignVariableOp4default_policy/default_policy/conv_value_1/bias/AdamFdefault_policy/default_policy/conv_value_1/bias/Adam/Initializer/zeros*
dtype0
?
Hdefault_policy/default_policy/conv_value_1/bias/Adam/Read/ReadVariableOpReadVariableOp4default_policy/default_policy/conv_value_1/bias/Adam*3
_class)
'%loc:@default_policy/conv_value_1/bias*
_output_shapes
:*
dtype0
?
Hdefault_policy/default_policy/conv_value_1/bias/Adam_1/Initializer/zerosConst*3
_class)
'%loc:@default_policy/conv_value_1/bias*
_output_shapes
:*
dtype0*
valueB*    
?
6default_policy/default_policy/conv_value_1/bias/Adam_1VarHandleOp*3
_class)
'%loc:@default_policy/conv_value_1/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*G
shared_name86default_policy/default_policy/conv_value_1/bias/Adam_1
?
Wdefault_policy/default_policy/conv_value_1/bias/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp6default_policy/default_policy/conv_value_1/bias/Adam_1*3
_class)
'%loc:@default_policy/conv_value_1/bias*
_output_shapes
: 
?
=default_policy/default_policy/conv_value_1/bias/Adam_1/AssignAssignVariableOp6default_policy/default_policy/conv_value_1/bias/Adam_1Hdefault_policy/default_policy/conv_value_1/bias/Adam_1/Initializer/zeros*
dtype0
?
Jdefault_policy/default_policy/conv_value_1/bias/Adam_1/Read/ReadVariableOpReadVariableOp6default_policy/default_policy/conv_value_1/bias/Adam_1*3
_class)
'%loc:@default_policy/conv_value_1/bias*
_output_shapes
:*
dtype0
?
Qdefault_policy/default_policy/conv1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@default_policy/conv1/kernel*
_output_shapes
:*
dtype0*%
valueB"            
?
Gdefault_policy/default_policy/conv1/kernel/Adam/Initializer/zeros/ConstConst*.
_class$
" loc:@default_policy/conv1/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
?
Adefault_policy/default_policy/conv1/kernel/Adam/Initializer/zerosFillQdefault_policy/default_policy/conv1/kernel/Adam/Initializer/zeros/shape_as_tensorGdefault_policy/default_policy/conv1/kernel/Adam/Initializer/zeros/Const*
T0*.
_class$
" loc:@default_policy/conv1/kernel*&
_output_shapes
:*

index_type0
?
/default_policy/default_policy/conv1/kernel/AdamVarHandleOp*.
_class$
" loc:@default_policy/conv1/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*@
shared_name1/default_policy/default_policy/conv1/kernel/Adam
?
Pdefault_policy/default_policy/conv1/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp/default_policy/default_policy/conv1/kernel/Adam*.
_class$
" loc:@default_policy/conv1/kernel*
_output_shapes
: 
?
6default_policy/default_policy/conv1/kernel/Adam/AssignAssignVariableOp/default_policy/default_policy/conv1/kernel/AdamAdefault_policy/default_policy/conv1/kernel/Adam/Initializer/zeros*
dtype0
?
Cdefault_policy/default_policy/conv1/kernel/Adam/Read/ReadVariableOpReadVariableOp/default_policy/default_policy/conv1/kernel/Adam*.
_class$
" loc:@default_policy/conv1/kernel*&
_output_shapes
:*
dtype0
?
Sdefault_policy/default_policy/conv1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@default_policy/conv1/kernel*
_output_shapes
:*
dtype0*%
valueB"            
?
Idefault_policy/default_policy/conv1/kernel/Adam_1/Initializer/zeros/ConstConst*.
_class$
" loc:@default_policy/conv1/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
?
Cdefault_policy/default_policy/conv1/kernel/Adam_1/Initializer/zerosFillSdefault_policy/default_policy/conv1/kernel/Adam_1/Initializer/zeros/shape_as_tensorIdefault_policy/default_policy/conv1/kernel/Adam_1/Initializer/zeros/Const*
T0*.
_class$
" loc:@default_policy/conv1/kernel*&
_output_shapes
:*

index_type0
?
1default_policy/default_policy/conv1/kernel/Adam_1VarHandleOp*.
_class$
" loc:@default_policy/conv1/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*B
shared_name31default_policy/default_policy/conv1/kernel/Adam_1
?
Rdefault_policy/default_policy/conv1/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp1default_policy/default_policy/conv1/kernel/Adam_1*.
_class$
" loc:@default_policy/conv1/kernel*
_output_shapes
: 
?
8default_policy/default_policy/conv1/kernel/Adam_1/AssignAssignVariableOp1default_policy/default_policy/conv1/kernel/Adam_1Cdefault_policy/default_policy/conv1/kernel/Adam_1/Initializer/zeros*
dtype0
?
Edefault_policy/default_policy/conv1/kernel/Adam_1/Read/ReadVariableOpReadVariableOp1default_policy/default_policy/conv1/kernel/Adam_1*.
_class$
" loc:@default_policy/conv1/kernel*&
_output_shapes
:*
dtype0
?
?default_policy/default_policy/conv1/bias/Adam/Initializer/zerosConst*,
_class"
 loc:@default_policy/conv1/bias*
_output_shapes
:*
dtype0*
valueB*    
?
-default_policy/default_policy/conv1/bias/AdamVarHandleOp*,
_class"
 loc:@default_policy/conv1/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*>
shared_name/-default_policy/default_policy/conv1/bias/Adam
?
Ndefault_policy/default_policy/conv1/bias/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp-default_policy/default_policy/conv1/bias/Adam*,
_class"
 loc:@default_policy/conv1/bias*
_output_shapes
: 
?
4default_policy/default_policy/conv1/bias/Adam/AssignAssignVariableOp-default_policy/default_policy/conv1/bias/Adam?default_policy/default_policy/conv1/bias/Adam/Initializer/zeros*
dtype0
?
Adefault_policy/default_policy/conv1/bias/Adam/Read/ReadVariableOpReadVariableOp-default_policy/default_policy/conv1/bias/Adam*,
_class"
 loc:@default_policy/conv1/bias*
_output_shapes
:*
dtype0
?
Adefault_policy/default_policy/conv1/bias/Adam_1/Initializer/zerosConst*,
_class"
 loc:@default_policy/conv1/bias*
_output_shapes
:*
dtype0*
valueB*    
?
/default_policy/default_policy/conv1/bias/Adam_1VarHandleOp*,
_class"
 loc:@default_policy/conv1/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*@
shared_name1/default_policy/default_policy/conv1/bias/Adam_1
?
Pdefault_policy/default_policy/conv1/bias/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp/default_policy/default_policy/conv1/bias/Adam_1*,
_class"
 loc:@default_policy/conv1/bias*
_output_shapes
: 
?
6default_policy/default_policy/conv1/bias/Adam_1/AssignAssignVariableOp/default_policy/default_policy/conv1/bias/Adam_1Adefault_policy/default_policy/conv1/bias/Adam_1/Initializer/zeros*
dtype0
?
Cdefault_policy/default_policy/conv1/bias/Adam_1/Read/ReadVariableOpReadVariableOp/default_policy/default_policy/conv1/bias/Adam_1*,
_class"
 loc:@default_policy/conv1/bias*
_output_shapes
:*
dtype0
?
Xdefault_policy/default_policy/conv_value_2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*5
_class+
)'loc:@default_policy/conv_value_2/kernel*
_output_shapes
:*
dtype0*%
valueB"             
?
Ndefault_policy/default_policy/conv_value_2/kernel/Adam/Initializer/zeros/ConstConst*5
_class+
)'loc:@default_policy/conv_value_2/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
?
Hdefault_policy/default_policy/conv_value_2/kernel/Adam/Initializer/zerosFillXdefault_policy/default_policy/conv_value_2/kernel/Adam/Initializer/zeros/shape_as_tensorNdefault_policy/default_policy/conv_value_2/kernel/Adam/Initializer/zeros/Const*
T0*5
_class+
)'loc:@default_policy/conv_value_2/kernel*&
_output_shapes
: *

index_type0
?
6default_policy/default_policy/conv_value_2/kernel/AdamVarHandleOp*5
_class+
)'loc:@default_policy/conv_value_2/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *G
shared_name86default_policy/default_policy/conv_value_2/kernel/Adam
?
Wdefault_policy/default_policy/conv_value_2/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp6default_policy/default_policy/conv_value_2/kernel/Adam*5
_class+
)'loc:@default_policy/conv_value_2/kernel*
_output_shapes
: 
?
=default_policy/default_policy/conv_value_2/kernel/Adam/AssignAssignVariableOp6default_policy/default_policy/conv_value_2/kernel/AdamHdefault_policy/default_policy/conv_value_2/kernel/Adam/Initializer/zeros*
dtype0
?
Jdefault_policy/default_policy/conv_value_2/kernel/Adam/Read/ReadVariableOpReadVariableOp6default_policy/default_policy/conv_value_2/kernel/Adam*5
_class+
)'loc:@default_policy/conv_value_2/kernel*&
_output_shapes
: *
dtype0
?
Zdefault_policy/default_policy/conv_value_2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*5
_class+
)'loc:@default_policy/conv_value_2/kernel*
_output_shapes
:*
dtype0*%
valueB"             
?
Pdefault_policy/default_policy/conv_value_2/kernel/Adam_1/Initializer/zeros/ConstConst*5
_class+
)'loc:@default_policy/conv_value_2/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
?
Jdefault_policy/default_policy/conv_value_2/kernel/Adam_1/Initializer/zerosFillZdefault_policy/default_policy/conv_value_2/kernel/Adam_1/Initializer/zeros/shape_as_tensorPdefault_policy/default_policy/conv_value_2/kernel/Adam_1/Initializer/zeros/Const*
T0*5
_class+
)'loc:@default_policy/conv_value_2/kernel*&
_output_shapes
: *

index_type0
?
8default_policy/default_policy/conv_value_2/kernel/Adam_1VarHandleOp*5
_class+
)'loc:@default_policy/conv_value_2/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *I
shared_name:8default_policy/default_policy/conv_value_2/kernel/Adam_1
?
Ydefault_policy/default_policy/conv_value_2/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp8default_policy/default_policy/conv_value_2/kernel/Adam_1*5
_class+
)'loc:@default_policy/conv_value_2/kernel*
_output_shapes
: 
?
?default_policy/default_policy/conv_value_2/kernel/Adam_1/AssignAssignVariableOp8default_policy/default_policy/conv_value_2/kernel/Adam_1Jdefault_policy/default_policy/conv_value_2/kernel/Adam_1/Initializer/zeros*
dtype0
?
Ldefault_policy/default_policy/conv_value_2/kernel/Adam_1/Read/ReadVariableOpReadVariableOp8default_policy/default_policy/conv_value_2/kernel/Adam_1*5
_class+
)'loc:@default_policy/conv_value_2/kernel*&
_output_shapes
: *
dtype0
?
Fdefault_policy/default_policy/conv_value_2/bias/Adam/Initializer/zerosConst*3
_class)
'%loc:@default_policy/conv_value_2/bias*
_output_shapes
: *
dtype0*
valueB *    
?
4default_policy/default_policy/conv_value_2/bias/AdamVarHandleOp*3
_class)
'%loc:@default_policy/conv_value_2/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *E
shared_name64default_policy/default_policy/conv_value_2/bias/Adam
?
Udefault_policy/default_policy/conv_value_2/bias/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp4default_policy/default_policy/conv_value_2/bias/Adam*3
_class)
'%loc:@default_policy/conv_value_2/bias*
_output_shapes
: 
?
;default_policy/default_policy/conv_value_2/bias/Adam/AssignAssignVariableOp4default_policy/default_policy/conv_value_2/bias/AdamFdefault_policy/default_policy/conv_value_2/bias/Adam/Initializer/zeros*
dtype0
?
Hdefault_policy/default_policy/conv_value_2/bias/Adam/Read/ReadVariableOpReadVariableOp4default_policy/default_policy/conv_value_2/bias/Adam*3
_class)
'%loc:@default_policy/conv_value_2/bias*
_output_shapes
: *
dtype0
?
Hdefault_policy/default_policy/conv_value_2/bias/Adam_1/Initializer/zerosConst*3
_class)
'%loc:@default_policy/conv_value_2/bias*
_output_shapes
: *
dtype0*
valueB *    
?
6default_policy/default_policy/conv_value_2/bias/Adam_1VarHandleOp*3
_class)
'%loc:@default_policy/conv_value_2/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *G
shared_name86default_policy/default_policy/conv_value_2/bias/Adam_1
?
Wdefault_policy/default_policy/conv_value_2/bias/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp6default_policy/default_policy/conv_value_2/bias/Adam_1*3
_class)
'%loc:@default_policy/conv_value_2/bias*
_output_shapes
: 
?
=default_policy/default_policy/conv_value_2/bias/Adam_1/AssignAssignVariableOp6default_policy/default_policy/conv_value_2/bias/Adam_1Hdefault_policy/default_policy/conv_value_2/bias/Adam_1/Initializer/zeros*
dtype0
?
Jdefault_policy/default_policy/conv_value_2/bias/Adam_1/Read/ReadVariableOpReadVariableOp6default_policy/default_policy/conv_value_2/bias/Adam_1*3
_class)
'%loc:@default_policy/conv_value_2/bias*
_output_shapes
: *
dtype0
?
Qdefault_policy/default_policy/conv2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@default_policy/conv2/kernel*
_output_shapes
:*
dtype0*%
valueB"             
?
Gdefault_policy/default_policy/conv2/kernel/Adam/Initializer/zeros/ConstConst*.
_class$
" loc:@default_policy/conv2/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
?
Adefault_policy/default_policy/conv2/kernel/Adam/Initializer/zerosFillQdefault_policy/default_policy/conv2/kernel/Adam/Initializer/zeros/shape_as_tensorGdefault_policy/default_policy/conv2/kernel/Adam/Initializer/zeros/Const*
T0*.
_class$
" loc:@default_policy/conv2/kernel*&
_output_shapes
: *

index_type0
?
/default_policy/default_policy/conv2/kernel/AdamVarHandleOp*.
_class$
" loc:@default_policy/conv2/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *@
shared_name1/default_policy/default_policy/conv2/kernel/Adam
?
Pdefault_policy/default_policy/conv2/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp/default_policy/default_policy/conv2/kernel/Adam*.
_class$
" loc:@default_policy/conv2/kernel*
_output_shapes
: 
?
6default_policy/default_policy/conv2/kernel/Adam/AssignAssignVariableOp/default_policy/default_policy/conv2/kernel/AdamAdefault_policy/default_policy/conv2/kernel/Adam/Initializer/zeros*
dtype0
?
Cdefault_policy/default_policy/conv2/kernel/Adam/Read/ReadVariableOpReadVariableOp/default_policy/default_policy/conv2/kernel/Adam*.
_class$
" loc:@default_policy/conv2/kernel*&
_output_shapes
: *
dtype0
?
Sdefault_policy/default_policy/conv2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@default_policy/conv2/kernel*
_output_shapes
:*
dtype0*%
valueB"             
?
Idefault_policy/default_policy/conv2/kernel/Adam_1/Initializer/zeros/ConstConst*.
_class$
" loc:@default_policy/conv2/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
?
Cdefault_policy/default_policy/conv2/kernel/Adam_1/Initializer/zerosFillSdefault_policy/default_policy/conv2/kernel/Adam_1/Initializer/zeros/shape_as_tensorIdefault_policy/default_policy/conv2/kernel/Adam_1/Initializer/zeros/Const*
T0*.
_class$
" loc:@default_policy/conv2/kernel*&
_output_shapes
: *

index_type0
?
1default_policy/default_policy/conv2/kernel/Adam_1VarHandleOp*.
_class$
" loc:@default_policy/conv2/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *B
shared_name31default_policy/default_policy/conv2/kernel/Adam_1
?
Rdefault_policy/default_policy/conv2/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp1default_policy/default_policy/conv2/kernel/Adam_1*.
_class$
" loc:@default_policy/conv2/kernel*
_output_shapes
: 
?
8default_policy/default_policy/conv2/kernel/Adam_1/AssignAssignVariableOp1default_policy/default_policy/conv2/kernel/Adam_1Cdefault_policy/default_policy/conv2/kernel/Adam_1/Initializer/zeros*
dtype0
?
Edefault_policy/default_policy/conv2/kernel/Adam_1/Read/ReadVariableOpReadVariableOp1default_policy/default_policy/conv2/kernel/Adam_1*.
_class$
" loc:@default_policy/conv2/kernel*&
_output_shapes
: *
dtype0
?
?default_policy/default_policy/conv2/bias/Adam/Initializer/zerosConst*,
_class"
 loc:@default_policy/conv2/bias*
_output_shapes
: *
dtype0*
valueB *    
?
-default_policy/default_policy/conv2/bias/AdamVarHandleOp*,
_class"
 loc:@default_policy/conv2/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *>
shared_name/-default_policy/default_policy/conv2/bias/Adam
?
Ndefault_policy/default_policy/conv2/bias/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp-default_policy/default_policy/conv2/bias/Adam*,
_class"
 loc:@default_policy/conv2/bias*
_output_shapes
: 
?
4default_policy/default_policy/conv2/bias/Adam/AssignAssignVariableOp-default_policy/default_policy/conv2/bias/Adam?default_policy/default_policy/conv2/bias/Adam/Initializer/zeros*
dtype0
?
Adefault_policy/default_policy/conv2/bias/Adam/Read/ReadVariableOpReadVariableOp-default_policy/default_policy/conv2/bias/Adam*,
_class"
 loc:@default_policy/conv2/bias*
_output_shapes
: *
dtype0
?
Adefault_policy/default_policy/conv2/bias/Adam_1/Initializer/zerosConst*,
_class"
 loc:@default_policy/conv2/bias*
_output_shapes
: *
dtype0*
valueB *    
?
/default_policy/default_policy/conv2/bias/Adam_1VarHandleOp*,
_class"
 loc:@default_policy/conv2/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *@
shared_name1/default_policy/default_policy/conv2/bias/Adam_1
?
Pdefault_policy/default_policy/conv2/bias/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp/default_policy/default_policy/conv2/bias/Adam_1*,
_class"
 loc:@default_policy/conv2/bias*
_output_shapes
: 
?
6default_policy/default_policy/conv2/bias/Adam_1/AssignAssignVariableOp/default_policy/default_policy/conv2/bias/Adam_1Adefault_policy/default_policy/conv2/bias/Adam_1/Initializer/zeros*
dtype0
?
Cdefault_policy/default_policy/conv2/bias/Adam_1/Read/ReadVariableOpReadVariableOp/default_policy/default_policy/conv2/bias/Adam_1*,
_class"
 loc:@default_policy/conv2/bias*
_output_shapes
: *
dtype0
?
Xdefault_policy/default_policy/conv_value_3/kernel/Adam/Initializer/zeros/shape_as_tensorConst*5
_class+
)'loc:@default_policy/conv_value_3/kernel*
_output_shapes
:*
dtype0*%
valueB"             
?
Ndefault_policy/default_policy/conv_value_3/kernel/Adam/Initializer/zeros/ConstConst*5
_class+
)'loc:@default_policy/conv_value_3/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
?
Hdefault_policy/default_policy/conv_value_3/kernel/Adam/Initializer/zerosFillXdefault_policy/default_policy/conv_value_3/kernel/Adam/Initializer/zeros/shape_as_tensorNdefault_policy/default_policy/conv_value_3/kernel/Adam/Initializer/zeros/Const*
T0*5
_class+
)'loc:@default_policy/conv_value_3/kernel*'
_output_shapes
: ?*

index_type0
?
6default_policy/default_policy/conv_value_3/kernel/AdamVarHandleOp*5
_class+
)'loc:@default_policy/conv_value_3/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: ?*G
shared_name86default_policy/default_policy/conv_value_3/kernel/Adam
?
Wdefault_policy/default_policy/conv_value_3/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp6default_policy/default_policy/conv_value_3/kernel/Adam*5
_class+
)'loc:@default_policy/conv_value_3/kernel*
_output_shapes
: 
?
=default_policy/default_policy/conv_value_3/kernel/Adam/AssignAssignVariableOp6default_policy/default_policy/conv_value_3/kernel/AdamHdefault_policy/default_policy/conv_value_3/kernel/Adam/Initializer/zeros*
dtype0
?
Jdefault_policy/default_policy/conv_value_3/kernel/Adam/Read/ReadVariableOpReadVariableOp6default_policy/default_policy/conv_value_3/kernel/Adam*5
_class+
)'loc:@default_policy/conv_value_3/kernel*'
_output_shapes
: ?*
dtype0
?
Zdefault_policy/default_policy/conv_value_3/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*5
_class+
)'loc:@default_policy/conv_value_3/kernel*
_output_shapes
:*
dtype0*%
valueB"             
?
Pdefault_policy/default_policy/conv_value_3/kernel/Adam_1/Initializer/zeros/ConstConst*5
_class+
)'loc:@default_policy/conv_value_3/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
?
Jdefault_policy/default_policy/conv_value_3/kernel/Adam_1/Initializer/zerosFillZdefault_policy/default_policy/conv_value_3/kernel/Adam_1/Initializer/zeros/shape_as_tensorPdefault_policy/default_policy/conv_value_3/kernel/Adam_1/Initializer/zeros/Const*
T0*5
_class+
)'loc:@default_policy/conv_value_3/kernel*'
_output_shapes
: ?*

index_type0
?
8default_policy/default_policy/conv_value_3/kernel/Adam_1VarHandleOp*5
_class+
)'loc:@default_policy/conv_value_3/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: ?*I
shared_name:8default_policy/default_policy/conv_value_3/kernel/Adam_1
?
Ydefault_policy/default_policy/conv_value_3/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp8default_policy/default_policy/conv_value_3/kernel/Adam_1*5
_class+
)'loc:@default_policy/conv_value_3/kernel*
_output_shapes
: 
?
?default_policy/default_policy/conv_value_3/kernel/Adam_1/AssignAssignVariableOp8default_policy/default_policy/conv_value_3/kernel/Adam_1Jdefault_policy/default_policy/conv_value_3/kernel/Adam_1/Initializer/zeros*
dtype0
?
Ldefault_policy/default_policy/conv_value_3/kernel/Adam_1/Read/ReadVariableOpReadVariableOp8default_policy/default_policy/conv_value_3/kernel/Adam_1*5
_class+
)'loc:@default_policy/conv_value_3/kernel*'
_output_shapes
: ?*
dtype0
?
Fdefault_policy/default_policy/conv_value_3/bias/Adam/Initializer/zerosConst*3
_class)
'%loc:@default_policy/conv_value_3/bias*
_output_shapes	
:?*
dtype0*
valueB?*    
?
4default_policy/default_policy/conv_value_3/bias/AdamVarHandleOp*3
_class)
'%loc:@default_policy/conv_value_3/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:?*E
shared_name64default_policy/default_policy/conv_value_3/bias/Adam
?
Udefault_policy/default_policy/conv_value_3/bias/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp4default_policy/default_policy/conv_value_3/bias/Adam*3
_class)
'%loc:@default_policy/conv_value_3/bias*
_output_shapes
: 
?
;default_policy/default_policy/conv_value_3/bias/Adam/AssignAssignVariableOp4default_policy/default_policy/conv_value_3/bias/AdamFdefault_policy/default_policy/conv_value_3/bias/Adam/Initializer/zeros*
dtype0
?
Hdefault_policy/default_policy/conv_value_3/bias/Adam/Read/ReadVariableOpReadVariableOp4default_policy/default_policy/conv_value_3/bias/Adam*3
_class)
'%loc:@default_policy/conv_value_3/bias*
_output_shapes	
:?*
dtype0
?
Hdefault_policy/default_policy/conv_value_3/bias/Adam_1/Initializer/zerosConst*3
_class)
'%loc:@default_policy/conv_value_3/bias*
_output_shapes	
:?*
dtype0*
valueB?*    
?
6default_policy/default_policy/conv_value_3/bias/Adam_1VarHandleOp*3
_class)
'%loc:@default_policy/conv_value_3/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:?*G
shared_name86default_policy/default_policy/conv_value_3/bias/Adam_1
?
Wdefault_policy/default_policy/conv_value_3/bias/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp6default_policy/default_policy/conv_value_3/bias/Adam_1*3
_class)
'%loc:@default_policy/conv_value_3/bias*
_output_shapes
: 
?
=default_policy/default_policy/conv_value_3/bias/Adam_1/AssignAssignVariableOp6default_policy/default_policy/conv_value_3/bias/Adam_1Hdefault_policy/default_policy/conv_value_3/bias/Adam_1/Initializer/zeros*
dtype0
?
Jdefault_policy/default_policy/conv_value_3/bias/Adam_1/Read/ReadVariableOpReadVariableOp6default_policy/default_policy/conv_value_3/bias/Adam_1*3
_class)
'%loc:@default_policy/conv_value_3/bias*
_output_shapes	
:?*
dtype0
?
Qdefault_policy/default_policy/conv3/kernel/Adam/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@default_policy/conv3/kernel*
_output_shapes
:*
dtype0*%
valueB"             
?
Gdefault_policy/default_policy/conv3/kernel/Adam/Initializer/zeros/ConstConst*.
_class$
" loc:@default_policy/conv3/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
?
Adefault_policy/default_policy/conv3/kernel/Adam/Initializer/zerosFillQdefault_policy/default_policy/conv3/kernel/Adam/Initializer/zeros/shape_as_tensorGdefault_policy/default_policy/conv3/kernel/Adam/Initializer/zeros/Const*
T0*.
_class$
" loc:@default_policy/conv3/kernel*'
_output_shapes
: ?*

index_type0
?
/default_policy/default_policy/conv3/kernel/AdamVarHandleOp*.
_class$
" loc:@default_policy/conv3/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: ?*@
shared_name1/default_policy/default_policy/conv3/kernel/Adam
?
Pdefault_policy/default_policy/conv3/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp/default_policy/default_policy/conv3/kernel/Adam*.
_class$
" loc:@default_policy/conv3/kernel*
_output_shapes
: 
?
6default_policy/default_policy/conv3/kernel/Adam/AssignAssignVariableOp/default_policy/default_policy/conv3/kernel/AdamAdefault_policy/default_policy/conv3/kernel/Adam/Initializer/zeros*
dtype0
?
Cdefault_policy/default_policy/conv3/kernel/Adam/Read/ReadVariableOpReadVariableOp/default_policy/default_policy/conv3/kernel/Adam*.
_class$
" loc:@default_policy/conv3/kernel*'
_output_shapes
: ?*
dtype0
?
Sdefault_policy/default_policy/conv3/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@default_policy/conv3/kernel*
_output_shapes
:*
dtype0*%
valueB"             
?
Idefault_policy/default_policy/conv3/kernel/Adam_1/Initializer/zeros/ConstConst*.
_class$
" loc:@default_policy/conv3/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
?
Cdefault_policy/default_policy/conv3/kernel/Adam_1/Initializer/zerosFillSdefault_policy/default_policy/conv3/kernel/Adam_1/Initializer/zeros/shape_as_tensorIdefault_policy/default_policy/conv3/kernel/Adam_1/Initializer/zeros/Const*
T0*.
_class$
" loc:@default_policy/conv3/kernel*'
_output_shapes
: ?*

index_type0
?
1default_policy/default_policy/conv3/kernel/Adam_1VarHandleOp*.
_class$
" loc:@default_policy/conv3/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: ?*B
shared_name31default_policy/default_policy/conv3/kernel/Adam_1
?
Rdefault_policy/default_policy/conv3/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp1default_policy/default_policy/conv3/kernel/Adam_1*.
_class$
" loc:@default_policy/conv3/kernel*
_output_shapes
: 
?
8default_policy/default_policy/conv3/kernel/Adam_1/AssignAssignVariableOp1default_policy/default_policy/conv3/kernel/Adam_1Cdefault_policy/default_policy/conv3/kernel/Adam_1/Initializer/zeros*
dtype0
?
Edefault_policy/default_policy/conv3/kernel/Adam_1/Read/ReadVariableOpReadVariableOp1default_policy/default_policy/conv3/kernel/Adam_1*.
_class$
" loc:@default_policy/conv3/kernel*'
_output_shapes
: ?*
dtype0
?
?default_policy/default_policy/conv3/bias/Adam/Initializer/zerosConst*,
_class"
 loc:@default_policy/conv3/bias*
_output_shapes	
:?*
dtype0*
valueB?*    
?
-default_policy/default_policy/conv3/bias/AdamVarHandleOp*,
_class"
 loc:@default_policy/conv3/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:?*>
shared_name/-default_policy/default_policy/conv3/bias/Adam
?
Ndefault_policy/default_policy/conv3/bias/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp-default_policy/default_policy/conv3/bias/Adam*,
_class"
 loc:@default_policy/conv3/bias*
_output_shapes
: 
?
4default_policy/default_policy/conv3/bias/Adam/AssignAssignVariableOp-default_policy/default_policy/conv3/bias/Adam?default_policy/default_policy/conv3/bias/Adam/Initializer/zeros*
dtype0
?
Adefault_policy/default_policy/conv3/bias/Adam/Read/ReadVariableOpReadVariableOp-default_policy/default_policy/conv3/bias/Adam*,
_class"
 loc:@default_policy/conv3/bias*
_output_shapes	
:?*
dtype0
?
Adefault_policy/default_policy/conv3/bias/Adam_1/Initializer/zerosConst*,
_class"
 loc:@default_policy/conv3/bias*
_output_shapes	
:?*
dtype0*
valueB?*    
?
/default_policy/default_policy/conv3/bias/Adam_1VarHandleOp*,
_class"
 loc:@default_policy/conv3/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:?*@
shared_name1/default_policy/default_policy/conv3/bias/Adam_1
?
Pdefault_policy/default_policy/conv3/bias/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp/default_policy/default_policy/conv3/bias/Adam_1*,
_class"
 loc:@default_policy/conv3/bias*
_output_shapes
: 
?
6default_policy/default_policy/conv3/bias/Adam_1/AssignAssignVariableOp/default_policy/default_policy/conv3/bias/Adam_1Adefault_policy/default_policy/conv3/bias/Adam_1/Initializer/zeros*
dtype0
?
Cdefault_policy/default_policy/conv3/bias/Adam_1/Read/ReadVariableOpReadVariableOp/default_policy/default_policy/conv3/bias/Adam_1*,
_class"
 loc:@default_policy/conv3/bias*
_output_shapes	
:?*
dtype0
?
Jdefault_policy/default_policy/conv_value_out/kernel/Adam/Initializer/zerosConst*7
_class-
+)loc:@default_policy/conv_value_out/kernel*'
_output_shapes
:?*
dtype0*&
valueB?*    
?
8default_policy/default_policy/conv_value_out/kernel/AdamVarHandleOp*7
_class-
+)loc:@default_policy/conv_value_out/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:?*I
shared_name:8default_policy/default_policy/conv_value_out/kernel/Adam
?
Ydefault_policy/default_policy/conv_value_out/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp8default_policy/default_policy/conv_value_out/kernel/Adam*7
_class-
+)loc:@default_policy/conv_value_out/kernel*
_output_shapes
: 
?
?default_policy/default_policy/conv_value_out/kernel/Adam/AssignAssignVariableOp8default_policy/default_policy/conv_value_out/kernel/AdamJdefault_policy/default_policy/conv_value_out/kernel/Adam/Initializer/zeros*
dtype0
?
Ldefault_policy/default_policy/conv_value_out/kernel/Adam/Read/ReadVariableOpReadVariableOp8default_policy/default_policy/conv_value_out/kernel/Adam*7
_class-
+)loc:@default_policy/conv_value_out/kernel*'
_output_shapes
:?*
dtype0
?
Ldefault_policy/default_policy/conv_value_out/kernel/Adam_1/Initializer/zerosConst*7
_class-
+)loc:@default_policy/conv_value_out/kernel*'
_output_shapes
:?*
dtype0*&
valueB?*    
?
:default_policy/default_policy/conv_value_out/kernel/Adam_1VarHandleOp*7
_class-
+)loc:@default_policy/conv_value_out/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:?*K
shared_name<:default_policy/default_policy/conv_value_out/kernel/Adam_1
?
[default_policy/default_policy/conv_value_out/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp:default_policy/default_policy/conv_value_out/kernel/Adam_1*7
_class-
+)loc:@default_policy/conv_value_out/kernel*
_output_shapes
: 
?
Adefault_policy/default_policy/conv_value_out/kernel/Adam_1/AssignAssignVariableOp:default_policy/default_policy/conv_value_out/kernel/Adam_1Ldefault_policy/default_policy/conv_value_out/kernel/Adam_1/Initializer/zeros*
dtype0
?
Ndefault_policy/default_policy/conv_value_out/kernel/Adam_1/Read/ReadVariableOpReadVariableOp:default_policy/default_policy/conv_value_out/kernel/Adam_1*7
_class-
+)loc:@default_policy/conv_value_out/kernel*'
_output_shapes
:?*
dtype0
?
Hdefault_policy/default_policy/conv_value_out/bias/Adam/Initializer/zerosConst*5
_class+
)'loc:@default_policy/conv_value_out/bias*
_output_shapes
:*
dtype0*
valueB*    
?
6default_policy/default_policy/conv_value_out/bias/AdamVarHandleOp*5
_class+
)'loc:@default_policy/conv_value_out/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*G
shared_name86default_policy/default_policy/conv_value_out/bias/Adam
?
Wdefault_policy/default_policy/conv_value_out/bias/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp6default_policy/default_policy/conv_value_out/bias/Adam*5
_class+
)'loc:@default_policy/conv_value_out/bias*
_output_shapes
: 
?
=default_policy/default_policy/conv_value_out/bias/Adam/AssignAssignVariableOp6default_policy/default_policy/conv_value_out/bias/AdamHdefault_policy/default_policy/conv_value_out/bias/Adam/Initializer/zeros*
dtype0
?
Jdefault_policy/default_policy/conv_value_out/bias/Adam/Read/ReadVariableOpReadVariableOp6default_policy/default_policy/conv_value_out/bias/Adam*5
_class+
)'loc:@default_policy/conv_value_out/bias*
_output_shapes
:*
dtype0
?
Jdefault_policy/default_policy/conv_value_out/bias/Adam_1/Initializer/zerosConst*5
_class+
)'loc:@default_policy/conv_value_out/bias*
_output_shapes
:*
dtype0*
valueB*    
?
8default_policy/default_policy/conv_value_out/bias/Adam_1VarHandleOp*5
_class+
)'loc:@default_policy/conv_value_out/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*I
shared_name:8default_policy/default_policy/conv_value_out/bias/Adam_1
?
Ydefault_policy/default_policy/conv_value_out/bias/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp8default_policy/default_policy/conv_value_out/bias/Adam_1*5
_class+
)'loc:@default_policy/conv_value_out/bias*
_output_shapes
: 
?
?default_policy/default_policy/conv_value_out/bias/Adam_1/AssignAssignVariableOp8default_policy/default_policy/conv_value_out/bias/Adam_1Jdefault_policy/default_policy/conv_value_out/bias/Adam_1/Initializer/zeros*
dtype0
?
Ldefault_policy/default_policy/conv_value_out/bias/Adam_1/Read/ReadVariableOpReadVariableOp8default_policy/default_policy/conv_value_out/bias/Adam_1*5
_class+
)'loc:@default_policy/conv_value_out/bias*
_output_shapes
:*
dtype0
?
Tdefault_policy/default_policy/conv_out/kernel/Adam/Initializer/zeros/shape_as_tensorConst*1
_class'
%#loc:@default_policy/conv_out/kernel*
_output_shapes
:*
dtype0*%
valueB"            
?
Jdefault_policy/default_policy/conv_out/kernel/Adam/Initializer/zeros/ConstConst*1
_class'
%#loc:@default_policy/conv_out/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
?
Ddefault_policy/default_policy/conv_out/kernel/Adam/Initializer/zerosFillTdefault_policy/default_policy/conv_out/kernel/Adam/Initializer/zeros/shape_as_tensorJdefault_policy/default_policy/conv_out/kernel/Adam/Initializer/zeros/Const*
T0*1
_class'
%#loc:@default_policy/conv_out/kernel*'
_output_shapes
:?*

index_type0
?
2default_policy/default_policy/conv_out/kernel/AdamVarHandleOp*1
_class'
%#loc:@default_policy/conv_out/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:?*C
shared_name42default_policy/default_policy/conv_out/kernel/Adam
?
Sdefault_policy/default_policy/conv_out/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp2default_policy/default_policy/conv_out/kernel/Adam*1
_class'
%#loc:@default_policy/conv_out/kernel*
_output_shapes
: 
?
9default_policy/default_policy/conv_out/kernel/Adam/AssignAssignVariableOp2default_policy/default_policy/conv_out/kernel/AdamDdefault_policy/default_policy/conv_out/kernel/Adam/Initializer/zeros*
dtype0
?
Fdefault_policy/default_policy/conv_out/kernel/Adam/Read/ReadVariableOpReadVariableOp2default_policy/default_policy/conv_out/kernel/Adam*1
_class'
%#loc:@default_policy/conv_out/kernel*'
_output_shapes
:?*
dtype0
?
Vdefault_policy/default_policy/conv_out/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*1
_class'
%#loc:@default_policy/conv_out/kernel*
_output_shapes
:*
dtype0*%
valueB"            
?
Ldefault_policy/default_policy/conv_out/kernel/Adam_1/Initializer/zeros/ConstConst*1
_class'
%#loc:@default_policy/conv_out/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
?
Fdefault_policy/default_policy/conv_out/kernel/Adam_1/Initializer/zerosFillVdefault_policy/default_policy/conv_out/kernel/Adam_1/Initializer/zeros/shape_as_tensorLdefault_policy/default_policy/conv_out/kernel/Adam_1/Initializer/zeros/Const*
T0*1
_class'
%#loc:@default_policy/conv_out/kernel*'
_output_shapes
:?*

index_type0
?
4default_policy/default_policy/conv_out/kernel/Adam_1VarHandleOp*1
_class'
%#loc:@default_policy/conv_out/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:?*E
shared_name64default_policy/default_policy/conv_out/kernel/Adam_1
?
Udefault_policy/default_policy/conv_out/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp4default_policy/default_policy/conv_out/kernel/Adam_1*1
_class'
%#loc:@default_policy/conv_out/kernel*
_output_shapes
: 
?
;default_policy/default_policy/conv_out/kernel/Adam_1/AssignAssignVariableOp4default_policy/default_policy/conv_out/kernel/Adam_1Fdefault_policy/default_policy/conv_out/kernel/Adam_1/Initializer/zeros*
dtype0
?
Hdefault_policy/default_policy/conv_out/kernel/Adam_1/Read/ReadVariableOpReadVariableOp4default_policy/default_policy/conv_out/kernel/Adam_1*1
_class'
%#loc:@default_policy/conv_out/kernel*'
_output_shapes
:?*
dtype0
?
Bdefault_policy/default_policy/conv_out/bias/Adam/Initializer/zerosConst*/
_class%
#!loc:@default_policy/conv_out/bias*
_output_shapes
:*
dtype0*
valueB*    
?
0default_policy/default_policy/conv_out/bias/AdamVarHandleOp*/
_class%
#!loc:@default_policy/conv_out/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*A
shared_name20default_policy/default_policy/conv_out/bias/Adam
?
Qdefault_policy/default_policy/conv_out/bias/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp0default_policy/default_policy/conv_out/bias/Adam*/
_class%
#!loc:@default_policy/conv_out/bias*
_output_shapes
: 
?
7default_policy/default_policy/conv_out/bias/Adam/AssignAssignVariableOp0default_policy/default_policy/conv_out/bias/AdamBdefault_policy/default_policy/conv_out/bias/Adam/Initializer/zeros*
dtype0
?
Ddefault_policy/default_policy/conv_out/bias/Adam/Read/ReadVariableOpReadVariableOp0default_policy/default_policy/conv_out/bias/Adam*/
_class%
#!loc:@default_policy/conv_out/bias*
_output_shapes
:*
dtype0
?
Ddefault_policy/default_policy/conv_out/bias/Adam_1/Initializer/zerosConst*/
_class%
#!loc:@default_policy/conv_out/bias*
_output_shapes
:*
dtype0*
valueB*    
?
2default_policy/default_policy/conv_out/bias/Adam_1VarHandleOp*/
_class%
#!loc:@default_policy/conv_out/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*C
shared_name42default_policy/default_policy/conv_out/bias/Adam_1
?
Sdefault_policy/default_policy/conv_out/bias/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp2default_policy/default_policy/conv_out/bias/Adam_1*/
_class%
#!loc:@default_policy/conv_out/bias*
_output_shapes
: 
?
9default_policy/default_policy/conv_out/bias/Adam_1/AssignAssignVariableOp2default_policy/default_policy/conv_out/bias/Adam_1Ddefault_policy/default_policy/conv_out/bias/Adam_1/Initializer/zeros*
dtype0
?
Fdefault_policy/default_policy/conv_out/bias/Adam_1/Read/ReadVariableOpReadVariableOp2default_policy/default_policy/conv_out/bias/Adam_1*/
_class%
#!loc:@default_policy/conv_out/bias*
_output_shapes
:*
dtype0
l
"default_policy/Adam/ReadVariableOpReadVariableOpdefault_policy/lr*
_output_shapes
: *
dtype0
^
default_policy/Adam/beta1Const*
_output_shapes
: *
dtype0*
valueB
 *fff?
^
default_policy/Adam/beta2Const*
_output_shapes
: *
dtype0*
valueB
 *w??
`
default_policy/Adam/epsilonConst*
_output_shapes
: *
dtype0*
valueB
 *w?+2
?
^default_policy/Adam/update_default_policy/conv_value_1/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power*
_output_shapes
: *
dtype0
?
`default_policy/Adam/update_default_policy/conv_value_1/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power*
_output_shapes
: *
dtype0
?
Odefault_policy/Adam/update_default_policy/conv_value_1/kernel/ResourceApplyAdamResourceApplyAdam"default_policy/conv_value_1/kernel6default_policy/default_policy/conv_value_1/kernel/Adam8default_policy/default_policy/conv_value_1/kernel/Adam_1^default_policy/Adam/update_default_policy/conv_value_1/kernel/ResourceApplyAdam/ReadVariableOp`default_policy/Adam/update_default_policy/conv_value_1/kernel/ResourceApplyAdam/ReadVariableOp_1"default_policy/Adam/ReadVariableOpdefault_policy/Adam/beta1default_policy/Adam/beta2default_policy/Adam/epsiloncdefault_policy/gradients/default_policy/model_2/conv_value_1/Conv2D_grad/tuple/control_dependency_1*
T0*5
_class+
)'loc:@default_policy/conv_value_1/kernel*
use_locking( *
use_nesterov( 
?
\default_policy/Adam/update_default_policy/conv_value_1/bias/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power*
_output_shapes
: *
dtype0
?
^default_policy/Adam/update_default_policy/conv_value_1/bias/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power*
_output_shapes
: *
dtype0
?
Mdefault_policy/Adam/update_default_policy/conv_value_1/bias/ResourceApplyAdamResourceApplyAdam default_policy/conv_value_1/bias4default_policy/default_policy/conv_value_1/bias/Adam6default_policy/default_policy/conv_value_1/bias/Adam_1\default_policy/Adam/update_default_policy/conv_value_1/bias/ResourceApplyAdam/ReadVariableOp^default_policy/Adam/update_default_policy/conv_value_1/bias/ResourceApplyAdam/ReadVariableOp_1"default_policy/Adam/ReadVariableOpdefault_policy/Adam/beta1default_policy/Adam/beta2default_policy/Adam/epsilonddefault_policy/gradients/default_policy/model_2/conv_value_1/BiasAdd_grad/tuple/control_dependency_1*
T0*3
_class)
'%loc:@default_policy/conv_value_1/bias*
use_locking( *
use_nesterov( 
?
Wdefault_policy/Adam/update_default_policy/conv1/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power*
_output_shapes
: *
dtype0
?
Ydefault_policy/Adam/update_default_policy/conv1/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power*
_output_shapes
: *
dtype0
?
Hdefault_policy/Adam/update_default_policy/conv1/kernel/ResourceApplyAdamResourceApplyAdamdefault_policy/conv1/kernel/default_policy/default_policy/conv1/kernel/Adam1default_policy/default_policy/conv1/kernel/Adam_1Wdefault_policy/Adam/update_default_policy/conv1/kernel/ResourceApplyAdam/ReadVariableOpYdefault_policy/Adam/update_default_policy/conv1/kernel/ResourceApplyAdam/ReadVariableOp_1"default_policy/Adam/ReadVariableOpdefault_policy/Adam/beta1default_policy/Adam/beta2default_policy/Adam/epsilon\default_policy/gradients/default_policy/model_2/conv1/Conv2D_grad/tuple/control_dependency_1*
T0*.
_class$
" loc:@default_policy/conv1/kernel*
use_locking( *
use_nesterov( 
?
Udefault_policy/Adam/update_default_policy/conv1/bias/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power*
_output_shapes
: *
dtype0
?
Wdefault_policy/Adam/update_default_policy/conv1/bias/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power*
_output_shapes
: *
dtype0
?
Fdefault_policy/Adam/update_default_policy/conv1/bias/ResourceApplyAdamResourceApplyAdamdefault_policy/conv1/bias-default_policy/default_policy/conv1/bias/Adam/default_policy/default_policy/conv1/bias/Adam_1Udefault_policy/Adam/update_default_policy/conv1/bias/ResourceApplyAdam/ReadVariableOpWdefault_policy/Adam/update_default_policy/conv1/bias/ResourceApplyAdam/ReadVariableOp_1"default_policy/Adam/ReadVariableOpdefault_policy/Adam/beta1default_policy/Adam/beta2default_policy/Adam/epsilon]default_policy/gradients/default_policy/model_2/conv1/BiasAdd_grad/tuple/control_dependency_1*
T0*,
_class"
 loc:@default_policy/conv1/bias*
use_locking( *
use_nesterov( 
?
^default_policy/Adam/update_default_policy/conv_value_2/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power*
_output_shapes
: *
dtype0
?
`default_policy/Adam/update_default_policy/conv_value_2/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power*
_output_shapes
: *
dtype0
?
Odefault_policy/Adam/update_default_policy/conv_value_2/kernel/ResourceApplyAdamResourceApplyAdam"default_policy/conv_value_2/kernel6default_policy/default_policy/conv_value_2/kernel/Adam8default_policy/default_policy/conv_value_2/kernel/Adam_1^default_policy/Adam/update_default_policy/conv_value_2/kernel/ResourceApplyAdam/ReadVariableOp`default_policy/Adam/update_default_policy/conv_value_2/kernel/ResourceApplyAdam/ReadVariableOp_1"default_policy/Adam/ReadVariableOpdefault_policy/Adam/beta1default_policy/Adam/beta2default_policy/Adam/epsiloncdefault_policy/gradients/default_policy/model_2/conv_value_2/Conv2D_grad/tuple/control_dependency_1*
T0*5
_class+
)'loc:@default_policy/conv_value_2/kernel*
use_locking( *
use_nesterov( 
?
\default_policy/Adam/update_default_policy/conv_value_2/bias/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power*
_output_shapes
: *
dtype0
?
^default_policy/Adam/update_default_policy/conv_value_2/bias/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power*
_output_shapes
: *
dtype0
?
Mdefault_policy/Adam/update_default_policy/conv_value_2/bias/ResourceApplyAdamResourceApplyAdam default_policy/conv_value_2/bias4default_policy/default_policy/conv_value_2/bias/Adam6default_policy/default_policy/conv_value_2/bias/Adam_1\default_policy/Adam/update_default_policy/conv_value_2/bias/ResourceApplyAdam/ReadVariableOp^default_policy/Adam/update_default_policy/conv_value_2/bias/ResourceApplyAdam/ReadVariableOp_1"default_policy/Adam/ReadVariableOpdefault_policy/Adam/beta1default_policy/Adam/beta2default_policy/Adam/epsilonddefault_policy/gradients/default_policy/model_2/conv_value_2/BiasAdd_grad/tuple/control_dependency_1*
T0*3
_class)
'%loc:@default_policy/conv_value_2/bias*
use_locking( *
use_nesterov( 
?
Wdefault_policy/Adam/update_default_policy/conv2/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power*
_output_shapes
: *
dtype0
?
Ydefault_policy/Adam/update_default_policy/conv2/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power*
_output_shapes
: *
dtype0
?
Hdefault_policy/Adam/update_default_policy/conv2/kernel/ResourceApplyAdamResourceApplyAdamdefault_policy/conv2/kernel/default_policy/default_policy/conv2/kernel/Adam1default_policy/default_policy/conv2/kernel/Adam_1Wdefault_policy/Adam/update_default_policy/conv2/kernel/ResourceApplyAdam/ReadVariableOpYdefault_policy/Adam/update_default_policy/conv2/kernel/ResourceApplyAdam/ReadVariableOp_1"default_policy/Adam/ReadVariableOpdefault_policy/Adam/beta1default_policy/Adam/beta2default_policy/Adam/epsilon\default_policy/gradients/default_policy/model_2/conv2/Conv2D_grad/tuple/control_dependency_1*
T0*.
_class$
" loc:@default_policy/conv2/kernel*
use_locking( *
use_nesterov( 
?
Udefault_policy/Adam/update_default_policy/conv2/bias/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power*
_output_shapes
: *
dtype0
?
Wdefault_policy/Adam/update_default_policy/conv2/bias/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power*
_output_shapes
: *
dtype0
?
Fdefault_policy/Adam/update_default_policy/conv2/bias/ResourceApplyAdamResourceApplyAdamdefault_policy/conv2/bias-default_policy/default_policy/conv2/bias/Adam/default_policy/default_policy/conv2/bias/Adam_1Udefault_policy/Adam/update_default_policy/conv2/bias/ResourceApplyAdam/ReadVariableOpWdefault_policy/Adam/update_default_policy/conv2/bias/ResourceApplyAdam/ReadVariableOp_1"default_policy/Adam/ReadVariableOpdefault_policy/Adam/beta1default_policy/Adam/beta2default_policy/Adam/epsilon]default_policy/gradients/default_policy/model_2/conv2/BiasAdd_grad/tuple/control_dependency_1*
T0*,
_class"
 loc:@default_policy/conv2/bias*
use_locking( *
use_nesterov( 
?
^default_policy/Adam/update_default_policy/conv_value_3/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power*
_output_shapes
: *
dtype0
?
`default_policy/Adam/update_default_policy/conv_value_3/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power*
_output_shapes
: *
dtype0
?
Odefault_policy/Adam/update_default_policy/conv_value_3/kernel/ResourceApplyAdamResourceApplyAdam"default_policy/conv_value_3/kernel6default_policy/default_policy/conv_value_3/kernel/Adam8default_policy/default_policy/conv_value_3/kernel/Adam_1^default_policy/Adam/update_default_policy/conv_value_3/kernel/ResourceApplyAdam/ReadVariableOp`default_policy/Adam/update_default_policy/conv_value_3/kernel/ResourceApplyAdam/ReadVariableOp_1"default_policy/Adam/ReadVariableOpdefault_policy/Adam/beta1default_policy/Adam/beta2default_policy/Adam/epsiloncdefault_policy/gradients/default_policy/model_2/conv_value_3/Conv2D_grad/tuple/control_dependency_1*
T0*5
_class+
)'loc:@default_policy/conv_value_3/kernel*
use_locking( *
use_nesterov( 
?
\default_policy/Adam/update_default_policy/conv_value_3/bias/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power*
_output_shapes
: *
dtype0
?
^default_policy/Adam/update_default_policy/conv_value_3/bias/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power*
_output_shapes
: *
dtype0
?
Mdefault_policy/Adam/update_default_policy/conv_value_3/bias/ResourceApplyAdamResourceApplyAdam default_policy/conv_value_3/bias4default_policy/default_policy/conv_value_3/bias/Adam6default_policy/default_policy/conv_value_3/bias/Adam_1\default_policy/Adam/update_default_policy/conv_value_3/bias/ResourceApplyAdam/ReadVariableOp^default_policy/Adam/update_default_policy/conv_value_3/bias/ResourceApplyAdam/ReadVariableOp_1"default_policy/Adam/ReadVariableOpdefault_policy/Adam/beta1default_policy/Adam/beta2default_policy/Adam/epsilonddefault_policy/gradients/default_policy/model_2/conv_value_3/BiasAdd_grad/tuple/control_dependency_1*
T0*3
_class)
'%loc:@default_policy/conv_value_3/bias*
use_locking( *
use_nesterov( 
?
Wdefault_policy/Adam/update_default_policy/conv3/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power*
_output_shapes
: *
dtype0
?
Ydefault_policy/Adam/update_default_policy/conv3/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power*
_output_shapes
: *
dtype0
?
Hdefault_policy/Adam/update_default_policy/conv3/kernel/ResourceApplyAdamResourceApplyAdamdefault_policy/conv3/kernel/default_policy/default_policy/conv3/kernel/Adam1default_policy/default_policy/conv3/kernel/Adam_1Wdefault_policy/Adam/update_default_policy/conv3/kernel/ResourceApplyAdam/ReadVariableOpYdefault_policy/Adam/update_default_policy/conv3/kernel/ResourceApplyAdam/ReadVariableOp_1"default_policy/Adam/ReadVariableOpdefault_policy/Adam/beta1default_policy/Adam/beta2default_policy/Adam/epsilon\default_policy/gradients/default_policy/model_2/conv3/Conv2D_grad/tuple/control_dependency_1*
T0*.
_class$
" loc:@default_policy/conv3/kernel*
use_locking( *
use_nesterov( 
?
Udefault_policy/Adam/update_default_policy/conv3/bias/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power*
_output_shapes
: *
dtype0
?
Wdefault_policy/Adam/update_default_policy/conv3/bias/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power*
_output_shapes
: *
dtype0
?
Fdefault_policy/Adam/update_default_policy/conv3/bias/ResourceApplyAdamResourceApplyAdamdefault_policy/conv3/bias-default_policy/default_policy/conv3/bias/Adam/default_policy/default_policy/conv3/bias/Adam_1Udefault_policy/Adam/update_default_policy/conv3/bias/ResourceApplyAdam/ReadVariableOpWdefault_policy/Adam/update_default_policy/conv3/bias/ResourceApplyAdam/ReadVariableOp_1"default_policy/Adam/ReadVariableOpdefault_policy/Adam/beta1default_policy/Adam/beta2default_policy/Adam/epsilon]default_policy/gradients/default_policy/model_2/conv3/BiasAdd_grad/tuple/control_dependency_1*
T0*,
_class"
 loc:@default_policy/conv3/bias*
use_locking( *
use_nesterov( 
?
`default_policy/Adam/update_default_policy/conv_value_out/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power*
_output_shapes
: *
dtype0
?
bdefault_policy/Adam/update_default_policy/conv_value_out/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power*
_output_shapes
: *
dtype0
?
Qdefault_policy/Adam/update_default_policy/conv_value_out/kernel/ResourceApplyAdamResourceApplyAdam$default_policy/conv_value_out/kernel8default_policy/default_policy/conv_value_out/kernel/Adam:default_policy/default_policy/conv_value_out/kernel/Adam_1`default_policy/Adam/update_default_policy/conv_value_out/kernel/ResourceApplyAdam/ReadVariableOpbdefault_policy/Adam/update_default_policy/conv_value_out/kernel/ResourceApplyAdam/ReadVariableOp_1"default_policy/Adam/ReadVariableOpdefault_policy/Adam/beta1default_policy/Adam/beta2default_policy/Adam/epsilonedefault_policy/gradients/default_policy/model_2/conv_value_out/Conv2D_grad/tuple/control_dependency_1*
T0*7
_class-
+)loc:@default_policy/conv_value_out/kernel*
use_locking( *
use_nesterov( 
?
^default_policy/Adam/update_default_policy/conv_value_out/bias/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power*
_output_shapes
: *
dtype0
?
`default_policy/Adam/update_default_policy/conv_value_out/bias/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power*
_output_shapes
: *
dtype0
?
Odefault_policy/Adam/update_default_policy/conv_value_out/bias/ResourceApplyAdamResourceApplyAdam"default_policy/conv_value_out/bias6default_policy/default_policy/conv_value_out/bias/Adam8default_policy/default_policy/conv_value_out/bias/Adam_1^default_policy/Adam/update_default_policy/conv_value_out/bias/ResourceApplyAdam/ReadVariableOp`default_policy/Adam/update_default_policy/conv_value_out/bias/ResourceApplyAdam/ReadVariableOp_1"default_policy/Adam/ReadVariableOpdefault_policy/Adam/beta1default_policy/Adam/beta2default_policy/Adam/epsilonfdefault_policy/gradients/default_policy/model_2/conv_value_out/BiasAdd_grad/tuple/control_dependency_1*
T0*5
_class+
)'loc:@default_policy/conv_value_out/bias*
use_locking( *
use_nesterov( 
?
Zdefault_policy/Adam/update_default_policy/conv_out/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power*
_output_shapes
: *
dtype0
?
\default_policy/Adam/update_default_policy/conv_out/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power*
_output_shapes
: *
dtype0
?
Kdefault_policy/Adam/update_default_policy/conv_out/kernel/ResourceApplyAdamResourceApplyAdamdefault_policy/conv_out/kernel2default_policy/default_policy/conv_out/kernel/Adam4default_policy/default_policy/conv_out/kernel/Adam_1Zdefault_policy/Adam/update_default_policy/conv_out/kernel/ResourceApplyAdam/ReadVariableOp\default_policy/Adam/update_default_policy/conv_out/kernel/ResourceApplyAdam/ReadVariableOp_1"default_policy/Adam/ReadVariableOpdefault_policy/Adam/beta1default_policy/Adam/beta2default_policy/Adam/epsilon_default_policy/gradients/default_policy/model_2/conv_out/Conv2D_grad/tuple/control_dependency_1*
T0*1
_class'
%#loc:@default_policy/conv_out/kernel*
use_locking( *
use_nesterov( 
?
Xdefault_policy/Adam/update_default_policy/conv_out/bias/ResourceApplyAdam/ReadVariableOpReadVariableOpdefault_policy/beta1_power*
_output_shapes
: *
dtype0
?
Zdefault_policy/Adam/update_default_policy/conv_out/bias/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpdefault_policy/beta2_power*
_output_shapes
: *
dtype0
?
Idefault_policy/Adam/update_default_policy/conv_out/bias/ResourceApplyAdamResourceApplyAdamdefault_policy/conv_out/bias0default_policy/default_policy/conv_out/bias/Adam2default_policy/default_policy/conv_out/bias/Adam_1Xdefault_policy/Adam/update_default_policy/conv_out/bias/ResourceApplyAdam/ReadVariableOpZdefault_policy/Adam/update_default_policy/conv_out/bias/ResourceApplyAdam/ReadVariableOp_1"default_policy/Adam/ReadVariableOpdefault_policy/Adam/beta1default_policy/Adam/beta2default_policy/Adam/epsilon`default_policy/gradients/default_policy/model_2/conv_out/BiasAdd_grad/tuple/control_dependency_1*
T0*/
_class%
#!loc:@default_policy/conv_out/bias*
use_locking( *
use_nesterov( 
?

$default_policy/Adam/ReadVariableOp_1ReadVariableOpdefault_policy/beta1_powerG^default_policy/Adam/update_default_policy/conv1/bias/ResourceApplyAdamI^default_policy/Adam/update_default_policy/conv1/kernel/ResourceApplyAdamG^default_policy/Adam/update_default_policy/conv2/bias/ResourceApplyAdamI^default_policy/Adam/update_default_policy/conv2/kernel/ResourceApplyAdamG^default_policy/Adam/update_default_policy/conv3/bias/ResourceApplyAdamI^default_policy/Adam/update_default_policy/conv3/kernel/ResourceApplyAdamJ^default_policy/Adam/update_default_policy/conv_out/bias/ResourceApplyAdamL^default_policy/Adam/update_default_policy/conv_out/kernel/ResourceApplyAdamN^default_policy/Adam/update_default_policy/conv_value_1/bias/ResourceApplyAdamP^default_policy/Adam/update_default_policy/conv_value_1/kernel/ResourceApplyAdamN^default_policy/Adam/update_default_policy/conv_value_2/bias/ResourceApplyAdamP^default_policy/Adam/update_default_policy/conv_value_2/kernel/ResourceApplyAdamN^default_policy/Adam/update_default_policy/conv_value_3/bias/ResourceApplyAdamP^default_policy/Adam/update_default_policy/conv_value_3/kernel/ResourceApplyAdamP^default_policy/Adam/update_default_policy/conv_value_out/bias/ResourceApplyAdamR^default_policy/Adam/update_default_policy/conv_value_out/kernel/ResourceApplyAdam*
_output_shapes
: *
dtype0
?
default_policy/Adam/mulMul$default_policy/Adam/ReadVariableOp_1default_policy/Adam/beta1*
T0*,
_class"
 loc:@default_policy/conv1/bias*
_output_shapes
: 
?
$default_policy/Adam/AssignVariableOpAssignVariableOpdefault_policy/beta1_powerdefault_policy/Adam/mul*,
_class"
 loc:@default_policy/conv1/bias*
dtype0
?
$default_policy/Adam/ReadVariableOp_2ReadVariableOpdefault_policy/beta1_power%^default_policy/Adam/AssignVariableOpG^default_policy/Adam/update_default_policy/conv1/bias/ResourceApplyAdamI^default_policy/Adam/update_default_policy/conv1/kernel/ResourceApplyAdamG^default_policy/Adam/update_default_policy/conv2/bias/ResourceApplyAdamI^default_policy/Adam/update_default_policy/conv2/kernel/ResourceApplyAdamG^default_policy/Adam/update_default_policy/conv3/bias/ResourceApplyAdamI^default_policy/Adam/update_default_policy/conv3/kernel/ResourceApplyAdamJ^default_policy/Adam/update_default_policy/conv_out/bias/ResourceApplyAdamL^default_policy/Adam/update_default_policy/conv_out/kernel/ResourceApplyAdamN^default_policy/Adam/update_default_policy/conv_value_1/bias/ResourceApplyAdamP^default_policy/Adam/update_default_policy/conv_value_1/kernel/ResourceApplyAdamN^default_policy/Adam/update_default_policy/conv_value_2/bias/ResourceApplyAdamP^default_policy/Adam/update_default_policy/conv_value_2/kernel/ResourceApplyAdamN^default_policy/Adam/update_default_policy/conv_value_3/bias/ResourceApplyAdamP^default_policy/Adam/update_default_policy/conv_value_3/kernel/ResourceApplyAdamP^default_policy/Adam/update_default_policy/conv_value_out/bias/ResourceApplyAdamR^default_policy/Adam/update_default_policy/conv_value_out/kernel/ResourceApplyAdam*,
_class"
 loc:@default_policy/conv1/bias*
_output_shapes
: *
dtype0
?

$default_policy/Adam/ReadVariableOp_3ReadVariableOpdefault_policy/beta2_powerG^default_policy/Adam/update_default_policy/conv1/bias/ResourceApplyAdamI^default_policy/Adam/update_default_policy/conv1/kernel/ResourceApplyAdamG^default_policy/Adam/update_default_policy/conv2/bias/ResourceApplyAdamI^default_policy/Adam/update_default_policy/conv2/kernel/ResourceApplyAdamG^default_policy/Adam/update_default_policy/conv3/bias/ResourceApplyAdamI^default_policy/Adam/update_default_policy/conv3/kernel/ResourceApplyAdamJ^default_policy/Adam/update_default_policy/conv_out/bias/ResourceApplyAdamL^default_policy/Adam/update_default_policy/conv_out/kernel/ResourceApplyAdamN^default_policy/Adam/update_default_policy/conv_value_1/bias/ResourceApplyAdamP^default_policy/Adam/update_default_policy/conv_value_1/kernel/ResourceApplyAdamN^default_policy/Adam/update_default_policy/conv_value_2/bias/ResourceApplyAdamP^default_policy/Adam/update_default_policy/conv_value_2/kernel/ResourceApplyAdamN^default_policy/Adam/update_default_policy/conv_value_3/bias/ResourceApplyAdamP^default_policy/Adam/update_default_policy/conv_value_3/kernel/ResourceApplyAdamP^default_policy/Adam/update_default_policy/conv_value_out/bias/ResourceApplyAdamR^default_policy/Adam/update_default_policy/conv_value_out/kernel/ResourceApplyAdam*
_output_shapes
: *
dtype0
?
default_policy/Adam/mul_1Mul$default_policy/Adam/ReadVariableOp_3default_policy/Adam/beta2*
T0*,
_class"
 loc:@default_policy/conv1/bias*
_output_shapes
: 
?
&default_policy/Adam/AssignVariableOp_1AssignVariableOpdefault_policy/beta2_powerdefault_policy/Adam/mul_1*,
_class"
 loc:@default_policy/conv1/bias*
dtype0
?
$default_policy/Adam/ReadVariableOp_4ReadVariableOpdefault_policy/beta2_power'^default_policy/Adam/AssignVariableOp_1G^default_policy/Adam/update_default_policy/conv1/bias/ResourceApplyAdamI^default_policy/Adam/update_default_policy/conv1/kernel/ResourceApplyAdamG^default_policy/Adam/update_default_policy/conv2/bias/ResourceApplyAdamI^default_policy/Adam/update_default_policy/conv2/kernel/ResourceApplyAdamG^default_policy/Adam/update_default_policy/conv3/bias/ResourceApplyAdamI^default_policy/Adam/update_default_policy/conv3/kernel/ResourceApplyAdamJ^default_policy/Adam/update_default_policy/conv_out/bias/ResourceApplyAdamL^default_policy/Adam/update_default_policy/conv_out/kernel/ResourceApplyAdamN^default_policy/Adam/update_default_policy/conv_value_1/bias/ResourceApplyAdamP^default_policy/Adam/update_default_policy/conv_value_1/kernel/ResourceApplyAdamN^default_policy/Adam/update_default_policy/conv_value_2/bias/ResourceApplyAdamP^default_policy/Adam/update_default_policy/conv_value_2/kernel/ResourceApplyAdamN^default_policy/Adam/update_default_policy/conv_value_3/bias/ResourceApplyAdamP^default_policy/Adam/update_default_policy/conv_value_3/kernel/ResourceApplyAdamP^default_policy/Adam/update_default_policy/conv_value_out/bias/ResourceApplyAdamR^default_policy/Adam/update_default_policy/conv_value_out/kernel/ResourceApplyAdam*,
_class"
 loc:@default_policy/conv1/bias*
_output_shapes
: *
dtype0
?

default_policy/Adam/updateNoOp%^default_policy/Adam/AssignVariableOp'^default_policy/Adam/AssignVariableOp_1G^default_policy/Adam/update_default_policy/conv1/bias/ResourceApplyAdamI^default_policy/Adam/update_default_policy/conv1/kernel/ResourceApplyAdamG^default_policy/Adam/update_default_policy/conv2/bias/ResourceApplyAdamI^default_policy/Adam/update_default_policy/conv2/kernel/ResourceApplyAdamG^default_policy/Adam/update_default_policy/conv3/bias/ResourceApplyAdamI^default_policy/Adam/update_default_policy/conv3/kernel/ResourceApplyAdamJ^default_policy/Adam/update_default_policy/conv_out/bias/ResourceApplyAdamL^default_policy/Adam/update_default_policy/conv_out/kernel/ResourceApplyAdamN^default_policy/Adam/update_default_policy/conv_value_1/bias/ResourceApplyAdamP^default_policy/Adam/update_default_policy/conv_value_1/kernel/ResourceApplyAdamN^default_policy/Adam/update_default_policy/conv_value_2/bias/ResourceApplyAdamP^default_policy/Adam/update_default_policy/conv_value_2/kernel/ResourceApplyAdamN^default_policy/Adam/update_default_policy/conv_value_3/bias/ResourceApplyAdamP^default_policy/Adam/update_default_policy/conv_value_3/kernel/ResourceApplyAdamP^default_policy/Adam/update_default_policy/conv_value_out/bias/ResourceApplyAdamR^default_policy/Adam/update_default_policy/conv_value_out/kernel/ResourceApplyAdam
?
default_policy/Adam/ConstConst^default_policy/Adam/update*-
_class#
!loc:@default_policy/global_step*
_output_shapes
: *
dtype0	*
value	B	 R
?
default_policy/AdamAssignAddVariableOpdefault_policy/global_stepdefault_policy/Adam/Const*-
_class#
!loc:@default_policy/global_step*
dtype0	
?
default_policy/init_1NoOp"^default_policy/beta1_power/Assign"^default_policy/beta2_power/Assign!^default_policy/conv1/bias/Assign#^default_policy/conv1/kernel/Assign!^default_policy/conv2/bias/Assign#^default_policy/conv2/kernel/Assign!^default_policy/conv3/bias/Assign#^default_policy/conv3/kernel/Assign$^default_policy/conv_out/bias/Assign&^default_policy/conv_out/kernel/Assign(^default_policy/conv_value_1/bias/Assign*^default_policy/conv_value_1/kernel/Assign(^default_policy/conv_value_2/bias/Assign*^default_policy/conv_value_2/kernel/Assign(^default_policy/conv_value_3/bias/Assign*^default_policy/conv_value_3/kernel/Assign*^default_policy/conv_value_out/bias/Assign,^default_policy/conv_value_out/kernel/Assign5^default_policy/default_policy/conv1/bias/Adam/Assign7^default_policy/default_policy/conv1/bias/Adam_1/Assign7^default_policy/default_policy/conv1/kernel/Adam/Assign9^default_policy/default_policy/conv1/kernel/Adam_1/Assign5^default_policy/default_policy/conv2/bias/Adam/Assign7^default_policy/default_policy/conv2/bias/Adam_1/Assign7^default_policy/default_policy/conv2/kernel/Adam/Assign9^default_policy/default_policy/conv2/kernel/Adam_1/Assign5^default_policy/default_policy/conv3/bias/Adam/Assign7^default_policy/default_policy/conv3/bias/Adam_1/Assign7^default_policy/default_policy/conv3/kernel/Adam/Assign9^default_policy/default_policy/conv3/kernel/Adam_1/Assign8^default_policy/default_policy/conv_out/bias/Adam/Assign:^default_policy/default_policy/conv_out/bias/Adam_1/Assign:^default_policy/default_policy/conv_out/kernel/Adam/Assign<^default_policy/default_policy/conv_out/kernel/Adam_1/Assign<^default_policy/default_policy/conv_value_1/bias/Adam/Assign>^default_policy/default_policy/conv_value_1/bias/Adam_1/Assign>^default_policy/default_policy/conv_value_1/kernel/Adam/Assign@^default_policy/default_policy/conv_value_1/kernel/Adam_1/Assign<^default_policy/default_policy/conv_value_2/bias/Adam/Assign>^default_policy/default_policy/conv_value_2/bias/Adam_1/Assign>^default_policy/default_policy/conv_value_2/kernel/Adam/Assign@^default_policy/default_policy/conv_value_2/kernel/Adam_1/Assign<^default_policy/default_policy/conv_value_3/bias/Adam/Assign>^default_policy/default_policy/conv_value_3/bias/Adam_1/Assign>^default_policy/default_policy/conv_value_3/kernel/Adam/Assign@^default_policy/default_policy/conv_value_3/kernel/Adam_1/Assign>^default_policy/default_policy/conv_value_out/bias/Adam/Assign@^default_policy/default_policy/conv_value_out/bias/Adam_1/Assign@^default_policy/default_policy/conv_value_out/kernel/Adam/AssignB^default_policy/default_policy/conv_value_out/kernel/Adam_1/Assign$^default_policy/entropy_coeff/Assign"^default_policy/global_step/Assign^default_policy/kl_coeff/Assign^default_policy/lr/Assign!^default_policy/timestep_1/Assign
s
 default_policy/ReadVariableOp_36ReadVariableOpdefault_policy/beta1_power*
_output_shapes
: *
dtype0
v
5default_policy/Placeholder_default_policy/beta1_powerPlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
"default_policy/AssignVariableOp_18AssignVariableOpdefault_policy/beta1_power5default_policy/Placeholder_default_policy/beta1_power*
dtype0
?
 default_policy/ReadVariableOp_37ReadVariableOpdefault_policy/beta1_power#^default_policy/AssignVariableOp_18*
_output_shapes
: *
dtype0
s
 default_policy/ReadVariableOp_38ReadVariableOpdefault_policy/beta2_power*
_output_shapes
: *
dtype0
v
5default_policy/Placeholder_default_policy/beta2_powerPlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
"default_policy/AssignVariableOp_19AssignVariableOpdefault_policy/beta2_power5default_policy/Placeholder_default_policy/beta2_power*
dtype0
?
 default_policy/ReadVariableOp_39ReadVariableOpdefault_policy/beta2_power#^default_policy/AssignVariableOp_19*
_output_shapes
: *
dtype0
?
 default_policy/ReadVariableOp_40ReadVariableOp6default_policy/default_policy/conv_value_1/kernel/Adam*&
_output_shapes
:*
dtype0
?
Qdefault_policy/Placeholder_default_policy/default_policy/conv_value_1/kernel/AdamPlaceholder*&
_output_shapes
:*
dtype0*
shape:
?
"default_policy/AssignVariableOp_20AssignVariableOp6default_policy/default_policy/conv_value_1/kernel/AdamQdefault_policy/Placeholder_default_policy/default_policy/conv_value_1/kernel/Adam*
dtype0
?
 default_policy/ReadVariableOp_41ReadVariableOp6default_policy/default_policy/conv_value_1/kernel/Adam#^default_policy/AssignVariableOp_20*&
_output_shapes
:*
dtype0
?
 default_policy/ReadVariableOp_42ReadVariableOp8default_policy/default_policy/conv_value_1/kernel/Adam_1*&
_output_shapes
:*
dtype0
?
Sdefault_policy/Placeholder_default_policy/default_policy/conv_value_1/kernel/Adam_1Placeholder*&
_output_shapes
:*
dtype0*
shape:
?
"default_policy/AssignVariableOp_21AssignVariableOp8default_policy/default_policy/conv_value_1/kernel/Adam_1Sdefault_policy/Placeholder_default_policy/default_policy/conv_value_1/kernel/Adam_1*
dtype0
?
 default_policy/ReadVariableOp_43ReadVariableOp8default_policy/default_policy/conv_value_1/kernel/Adam_1#^default_policy/AssignVariableOp_21*&
_output_shapes
:*
dtype0
?
 default_policy/ReadVariableOp_44ReadVariableOp4default_policy/default_policy/conv_value_1/bias/Adam*
_output_shapes
:*
dtype0
?
Odefault_policy/Placeholder_default_policy/default_policy/conv_value_1/bias/AdamPlaceholder*
_output_shapes
:*
dtype0*
shape:
?
"default_policy/AssignVariableOp_22AssignVariableOp4default_policy/default_policy/conv_value_1/bias/AdamOdefault_policy/Placeholder_default_policy/default_policy/conv_value_1/bias/Adam*
dtype0
?
 default_policy/ReadVariableOp_45ReadVariableOp4default_policy/default_policy/conv_value_1/bias/Adam#^default_policy/AssignVariableOp_22*
_output_shapes
:*
dtype0
?
 default_policy/ReadVariableOp_46ReadVariableOp6default_policy/default_policy/conv_value_1/bias/Adam_1*
_output_shapes
:*
dtype0
?
Qdefault_policy/Placeholder_default_policy/default_policy/conv_value_1/bias/Adam_1Placeholder*
_output_shapes
:*
dtype0*
shape:
?
"default_policy/AssignVariableOp_23AssignVariableOp6default_policy/default_policy/conv_value_1/bias/Adam_1Qdefault_policy/Placeholder_default_policy/default_policy/conv_value_1/bias/Adam_1*
dtype0
?
 default_policy/ReadVariableOp_47ReadVariableOp6default_policy/default_policy/conv_value_1/bias/Adam_1#^default_policy/AssignVariableOp_23*
_output_shapes
:*
dtype0
?
 default_policy/ReadVariableOp_48ReadVariableOp/default_policy/default_policy/conv1/kernel/Adam*&
_output_shapes
:*
dtype0
?
Jdefault_policy/Placeholder_default_policy/default_policy/conv1/kernel/AdamPlaceholder*&
_output_shapes
:*
dtype0*
shape:
?
"default_policy/AssignVariableOp_24AssignVariableOp/default_policy/default_policy/conv1/kernel/AdamJdefault_policy/Placeholder_default_policy/default_policy/conv1/kernel/Adam*
dtype0
?
 default_policy/ReadVariableOp_49ReadVariableOp/default_policy/default_policy/conv1/kernel/Adam#^default_policy/AssignVariableOp_24*&
_output_shapes
:*
dtype0
?
 default_policy/ReadVariableOp_50ReadVariableOp1default_policy/default_policy/conv1/kernel/Adam_1*&
_output_shapes
:*
dtype0
?
Ldefault_policy/Placeholder_default_policy/default_policy/conv1/kernel/Adam_1Placeholder*&
_output_shapes
:*
dtype0*
shape:
?
"default_policy/AssignVariableOp_25AssignVariableOp1default_policy/default_policy/conv1/kernel/Adam_1Ldefault_policy/Placeholder_default_policy/default_policy/conv1/kernel/Adam_1*
dtype0
?
 default_policy/ReadVariableOp_51ReadVariableOp1default_policy/default_policy/conv1/kernel/Adam_1#^default_policy/AssignVariableOp_25*&
_output_shapes
:*
dtype0
?
 default_policy/ReadVariableOp_52ReadVariableOp-default_policy/default_policy/conv1/bias/Adam*
_output_shapes
:*
dtype0
?
Hdefault_policy/Placeholder_default_policy/default_policy/conv1/bias/AdamPlaceholder*
_output_shapes
:*
dtype0*
shape:
?
"default_policy/AssignVariableOp_26AssignVariableOp-default_policy/default_policy/conv1/bias/AdamHdefault_policy/Placeholder_default_policy/default_policy/conv1/bias/Adam*
dtype0
?
 default_policy/ReadVariableOp_53ReadVariableOp-default_policy/default_policy/conv1/bias/Adam#^default_policy/AssignVariableOp_26*
_output_shapes
:*
dtype0
?
 default_policy/ReadVariableOp_54ReadVariableOp/default_policy/default_policy/conv1/bias/Adam_1*
_output_shapes
:*
dtype0
?
Jdefault_policy/Placeholder_default_policy/default_policy/conv1/bias/Adam_1Placeholder*
_output_shapes
:*
dtype0*
shape:
?
"default_policy/AssignVariableOp_27AssignVariableOp/default_policy/default_policy/conv1/bias/Adam_1Jdefault_policy/Placeholder_default_policy/default_policy/conv1/bias/Adam_1*
dtype0
?
 default_policy/ReadVariableOp_55ReadVariableOp/default_policy/default_policy/conv1/bias/Adam_1#^default_policy/AssignVariableOp_27*
_output_shapes
:*
dtype0
?
 default_policy/ReadVariableOp_56ReadVariableOp6default_policy/default_policy/conv_value_2/kernel/Adam*&
_output_shapes
: *
dtype0
?
Qdefault_policy/Placeholder_default_policy/default_policy/conv_value_2/kernel/AdamPlaceholder*&
_output_shapes
: *
dtype0*
shape: 
?
"default_policy/AssignVariableOp_28AssignVariableOp6default_policy/default_policy/conv_value_2/kernel/AdamQdefault_policy/Placeholder_default_policy/default_policy/conv_value_2/kernel/Adam*
dtype0
?
 default_policy/ReadVariableOp_57ReadVariableOp6default_policy/default_policy/conv_value_2/kernel/Adam#^default_policy/AssignVariableOp_28*&
_output_shapes
: *
dtype0
?
 default_policy/ReadVariableOp_58ReadVariableOp8default_policy/default_policy/conv_value_2/kernel/Adam_1*&
_output_shapes
: *
dtype0
?
Sdefault_policy/Placeholder_default_policy/default_policy/conv_value_2/kernel/Adam_1Placeholder*&
_output_shapes
: *
dtype0*
shape: 
?
"default_policy/AssignVariableOp_29AssignVariableOp8default_policy/default_policy/conv_value_2/kernel/Adam_1Sdefault_policy/Placeholder_default_policy/default_policy/conv_value_2/kernel/Adam_1*
dtype0
?
 default_policy/ReadVariableOp_59ReadVariableOp8default_policy/default_policy/conv_value_2/kernel/Adam_1#^default_policy/AssignVariableOp_29*&
_output_shapes
: *
dtype0
?
 default_policy/ReadVariableOp_60ReadVariableOp4default_policy/default_policy/conv_value_2/bias/Adam*
_output_shapes
: *
dtype0
?
Odefault_policy/Placeholder_default_policy/default_policy/conv_value_2/bias/AdamPlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
"default_policy/AssignVariableOp_30AssignVariableOp4default_policy/default_policy/conv_value_2/bias/AdamOdefault_policy/Placeholder_default_policy/default_policy/conv_value_2/bias/Adam*
dtype0
?
 default_policy/ReadVariableOp_61ReadVariableOp4default_policy/default_policy/conv_value_2/bias/Adam#^default_policy/AssignVariableOp_30*
_output_shapes
: *
dtype0
?
 default_policy/ReadVariableOp_62ReadVariableOp6default_policy/default_policy/conv_value_2/bias/Adam_1*
_output_shapes
: *
dtype0
?
Qdefault_policy/Placeholder_default_policy/default_policy/conv_value_2/bias/Adam_1Placeholder*
_output_shapes
: *
dtype0*
shape: 
?
"default_policy/AssignVariableOp_31AssignVariableOp6default_policy/default_policy/conv_value_2/bias/Adam_1Qdefault_policy/Placeholder_default_policy/default_policy/conv_value_2/bias/Adam_1*
dtype0
?
 default_policy/ReadVariableOp_63ReadVariableOp6default_policy/default_policy/conv_value_2/bias/Adam_1#^default_policy/AssignVariableOp_31*
_output_shapes
: *
dtype0
?
 default_policy/ReadVariableOp_64ReadVariableOp/default_policy/default_policy/conv2/kernel/Adam*&
_output_shapes
: *
dtype0
?
Jdefault_policy/Placeholder_default_policy/default_policy/conv2/kernel/AdamPlaceholder*&
_output_shapes
: *
dtype0*
shape: 
?
"default_policy/AssignVariableOp_32AssignVariableOp/default_policy/default_policy/conv2/kernel/AdamJdefault_policy/Placeholder_default_policy/default_policy/conv2/kernel/Adam*
dtype0
?
 default_policy/ReadVariableOp_65ReadVariableOp/default_policy/default_policy/conv2/kernel/Adam#^default_policy/AssignVariableOp_32*&
_output_shapes
: *
dtype0
?
 default_policy/ReadVariableOp_66ReadVariableOp1default_policy/default_policy/conv2/kernel/Adam_1*&
_output_shapes
: *
dtype0
?
Ldefault_policy/Placeholder_default_policy/default_policy/conv2/kernel/Adam_1Placeholder*&
_output_shapes
: *
dtype0*
shape: 
?
"default_policy/AssignVariableOp_33AssignVariableOp1default_policy/default_policy/conv2/kernel/Adam_1Ldefault_policy/Placeholder_default_policy/default_policy/conv2/kernel/Adam_1*
dtype0
?
 default_policy/ReadVariableOp_67ReadVariableOp1default_policy/default_policy/conv2/kernel/Adam_1#^default_policy/AssignVariableOp_33*&
_output_shapes
: *
dtype0
?
 default_policy/ReadVariableOp_68ReadVariableOp-default_policy/default_policy/conv2/bias/Adam*
_output_shapes
: *
dtype0
?
Hdefault_policy/Placeholder_default_policy/default_policy/conv2/bias/AdamPlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
"default_policy/AssignVariableOp_34AssignVariableOp-default_policy/default_policy/conv2/bias/AdamHdefault_policy/Placeholder_default_policy/default_policy/conv2/bias/Adam*
dtype0
?
 default_policy/ReadVariableOp_69ReadVariableOp-default_policy/default_policy/conv2/bias/Adam#^default_policy/AssignVariableOp_34*
_output_shapes
: *
dtype0
?
 default_policy/ReadVariableOp_70ReadVariableOp/default_policy/default_policy/conv2/bias/Adam_1*
_output_shapes
: *
dtype0
?
Jdefault_policy/Placeholder_default_policy/default_policy/conv2/bias/Adam_1Placeholder*
_output_shapes
: *
dtype0*
shape: 
?
"default_policy/AssignVariableOp_35AssignVariableOp/default_policy/default_policy/conv2/bias/Adam_1Jdefault_policy/Placeholder_default_policy/default_policy/conv2/bias/Adam_1*
dtype0
?
 default_policy/ReadVariableOp_71ReadVariableOp/default_policy/default_policy/conv2/bias/Adam_1#^default_policy/AssignVariableOp_35*
_output_shapes
: *
dtype0
?
 default_policy/ReadVariableOp_72ReadVariableOp6default_policy/default_policy/conv_value_3/kernel/Adam*'
_output_shapes
: ?*
dtype0
?
Qdefault_policy/Placeholder_default_policy/default_policy/conv_value_3/kernel/AdamPlaceholder*'
_output_shapes
: ?*
dtype0*
shape: ?
?
"default_policy/AssignVariableOp_36AssignVariableOp6default_policy/default_policy/conv_value_3/kernel/AdamQdefault_policy/Placeholder_default_policy/default_policy/conv_value_3/kernel/Adam*
dtype0
?
 default_policy/ReadVariableOp_73ReadVariableOp6default_policy/default_policy/conv_value_3/kernel/Adam#^default_policy/AssignVariableOp_36*'
_output_shapes
: ?*
dtype0
?
 default_policy/ReadVariableOp_74ReadVariableOp8default_policy/default_policy/conv_value_3/kernel/Adam_1*'
_output_shapes
: ?*
dtype0
?
Sdefault_policy/Placeholder_default_policy/default_policy/conv_value_3/kernel/Adam_1Placeholder*'
_output_shapes
: ?*
dtype0*
shape: ?
?
"default_policy/AssignVariableOp_37AssignVariableOp8default_policy/default_policy/conv_value_3/kernel/Adam_1Sdefault_policy/Placeholder_default_policy/default_policy/conv_value_3/kernel/Adam_1*
dtype0
?
 default_policy/ReadVariableOp_75ReadVariableOp8default_policy/default_policy/conv_value_3/kernel/Adam_1#^default_policy/AssignVariableOp_37*'
_output_shapes
: ?*
dtype0
?
 default_policy/ReadVariableOp_76ReadVariableOp4default_policy/default_policy/conv_value_3/bias/Adam*
_output_shapes	
:?*
dtype0
?
Odefault_policy/Placeholder_default_policy/default_policy/conv_value_3/bias/AdamPlaceholder*
_output_shapes	
:?*
dtype0*
shape:?
?
"default_policy/AssignVariableOp_38AssignVariableOp4default_policy/default_policy/conv_value_3/bias/AdamOdefault_policy/Placeholder_default_policy/default_policy/conv_value_3/bias/Adam*
dtype0
?
 default_policy/ReadVariableOp_77ReadVariableOp4default_policy/default_policy/conv_value_3/bias/Adam#^default_policy/AssignVariableOp_38*
_output_shapes	
:?*
dtype0
?
 default_policy/ReadVariableOp_78ReadVariableOp6default_policy/default_policy/conv_value_3/bias/Adam_1*
_output_shapes	
:?*
dtype0
?
Qdefault_policy/Placeholder_default_policy/default_policy/conv_value_3/bias/Adam_1Placeholder*
_output_shapes	
:?*
dtype0*
shape:?
?
"default_policy/AssignVariableOp_39AssignVariableOp6default_policy/default_policy/conv_value_3/bias/Adam_1Qdefault_policy/Placeholder_default_policy/default_policy/conv_value_3/bias/Adam_1*
dtype0
?
 default_policy/ReadVariableOp_79ReadVariableOp6default_policy/default_policy/conv_value_3/bias/Adam_1#^default_policy/AssignVariableOp_39*
_output_shapes	
:?*
dtype0
?
 default_policy/ReadVariableOp_80ReadVariableOp/default_policy/default_policy/conv3/kernel/Adam*'
_output_shapes
: ?*
dtype0
?
Jdefault_policy/Placeholder_default_policy/default_policy/conv3/kernel/AdamPlaceholder*'
_output_shapes
: ?*
dtype0*
shape: ?
?
"default_policy/AssignVariableOp_40AssignVariableOp/default_policy/default_policy/conv3/kernel/AdamJdefault_policy/Placeholder_default_policy/default_policy/conv3/kernel/Adam*
dtype0
?
 default_policy/ReadVariableOp_81ReadVariableOp/default_policy/default_policy/conv3/kernel/Adam#^default_policy/AssignVariableOp_40*'
_output_shapes
: ?*
dtype0
?
 default_policy/ReadVariableOp_82ReadVariableOp1default_policy/default_policy/conv3/kernel/Adam_1*'
_output_shapes
: ?*
dtype0
?
Ldefault_policy/Placeholder_default_policy/default_policy/conv3/kernel/Adam_1Placeholder*'
_output_shapes
: ?*
dtype0*
shape: ?
?
"default_policy/AssignVariableOp_41AssignVariableOp1default_policy/default_policy/conv3/kernel/Adam_1Ldefault_policy/Placeholder_default_policy/default_policy/conv3/kernel/Adam_1*
dtype0
?
 default_policy/ReadVariableOp_83ReadVariableOp1default_policy/default_policy/conv3/kernel/Adam_1#^default_policy/AssignVariableOp_41*'
_output_shapes
: ?*
dtype0
?
 default_policy/ReadVariableOp_84ReadVariableOp-default_policy/default_policy/conv3/bias/Adam*
_output_shapes	
:?*
dtype0
?
Hdefault_policy/Placeholder_default_policy/default_policy/conv3/bias/AdamPlaceholder*
_output_shapes	
:?*
dtype0*
shape:?
?
"default_policy/AssignVariableOp_42AssignVariableOp-default_policy/default_policy/conv3/bias/AdamHdefault_policy/Placeholder_default_policy/default_policy/conv3/bias/Adam*
dtype0
?
 default_policy/ReadVariableOp_85ReadVariableOp-default_policy/default_policy/conv3/bias/Adam#^default_policy/AssignVariableOp_42*
_output_shapes	
:?*
dtype0
?
 default_policy/ReadVariableOp_86ReadVariableOp/default_policy/default_policy/conv3/bias/Adam_1*
_output_shapes	
:?*
dtype0
?
Jdefault_policy/Placeholder_default_policy/default_policy/conv3/bias/Adam_1Placeholder*
_output_shapes	
:?*
dtype0*
shape:?
?
"default_policy/AssignVariableOp_43AssignVariableOp/default_policy/default_policy/conv3/bias/Adam_1Jdefault_policy/Placeholder_default_policy/default_policy/conv3/bias/Adam_1*
dtype0
?
 default_policy/ReadVariableOp_87ReadVariableOp/default_policy/default_policy/conv3/bias/Adam_1#^default_policy/AssignVariableOp_43*
_output_shapes	
:?*
dtype0
?
 default_policy/ReadVariableOp_88ReadVariableOp8default_policy/default_policy/conv_value_out/kernel/Adam*'
_output_shapes
:?*
dtype0
?
Sdefault_policy/Placeholder_default_policy/default_policy/conv_value_out/kernel/AdamPlaceholder*'
_output_shapes
:?*
dtype0*
shape:?
?
"default_policy/AssignVariableOp_44AssignVariableOp8default_policy/default_policy/conv_value_out/kernel/AdamSdefault_policy/Placeholder_default_policy/default_policy/conv_value_out/kernel/Adam*
dtype0
?
 default_policy/ReadVariableOp_89ReadVariableOp8default_policy/default_policy/conv_value_out/kernel/Adam#^default_policy/AssignVariableOp_44*'
_output_shapes
:?*
dtype0
?
 default_policy/ReadVariableOp_90ReadVariableOp:default_policy/default_policy/conv_value_out/kernel/Adam_1*'
_output_shapes
:?*
dtype0
?
Udefault_policy/Placeholder_default_policy/default_policy/conv_value_out/kernel/Adam_1Placeholder*'
_output_shapes
:?*
dtype0*
shape:?
?
"default_policy/AssignVariableOp_45AssignVariableOp:default_policy/default_policy/conv_value_out/kernel/Adam_1Udefault_policy/Placeholder_default_policy/default_policy/conv_value_out/kernel/Adam_1*
dtype0
?
 default_policy/ReadVariableOp_91ReadVariableOp:default_policy/default_policy/conv_value_out/kernel/Adam_1#^default_policy/AssignVariableOp_45*'
_output_shapes
:?*
dtype0
?
 default_policy/ReadVariableOp_92ReadVariableOp6default_policy/default_policy/conv_value_out/bias/Adam*
_output_shapes
:*
dtype0
?
Qdefault_policy/Placeholder_default_policy/default_policy/conv_value_out/bias/AdamPlaceholder*
_output_shapes
:*
dtype0*
shape:
?
"default_policy/AssignVariableOp_46AssignVariableOp6default_policy/default_policy/conv_value_out/bias/AdamQdefault_policy/Placeholder_default_policy/default_policy/conv_value_out/bias/Adam*
dtype0
?
 default_policy/ReadVariableOp_93ReadVariableOp6default_policy/default_policy/conv_value_out/bias/Adam#^default_policy/AssignVariableOp_46*
_output_shapes
:*
dtype0
?
 default_policy/ReadVariableOp_94ReadVariableOp8default_policy/default_policy/conv_value_out/bias/Adam_1*
_output_shapes
:*
dtype0
?
Sdefault_policy/Placeholder_default_policy/default_policy/conv_value_out/bias/Adam_1Placeholder*
_output_shapes
:*
dtype0*
shape:
?
"default_policy/AssignVariableOp_47AssignVariableOp8default_policy/default_policy/conv_value_out/bias/Adam_1Sdefault_policy/Placeholder_default_policy/default_policy/conv_value_out/bias/Adam_1*
dtype0
?
 default_policy/ReadVariableOp_95ReadVariableOp8default_policy/default_policy/conv_value_out/bias/Adam_1#^default_policy/AssignVariableOp_47*
_output_shapes
:*
dtype0
?
 default_policy/ReadVariableOp_96ReadVariableOp2default_policy/default_policy/conv_out/kernel/Adam*'
_output_shapes
:?*
dtype0
?
Mdefault_policy/Placeholder_default_policy/default_policy/conv_out/kernel/AdamPlaceholder*'
_output_shapes
:?*
dtype0*
shape:?
?
"default_policy/AssignVariableOp_48AssignVariableOp2default_policy/default_policy/conv_out/kernel/AdamMdefault_policy/Placeholder_default_policy/default_policy/conv_out/kernel/Adam*
dtype0
?
 default_policy/ReadVariableOp_97ReadVariableOp2default_policy/default_policy/conv_out/kernel/Adam#^default_policy/AssignVariableOp_48*'
_output_shapes
:?*
dtype0
?
 default_policy/ReadVariableOp_98ReadVariableOp4default_policy/default_policy/conv_out/kernel/Adam_1*'
_output_shapes
:?*
dtype0
?
Odefault_policy/Placeholder_default_policy/default_policy/conv_out/kernel/Adam_1Placeholder*'
_output_shapes
:?*
dtype0*
shape:?
?
"default_policy/AssignVariableOp_49AssignVariableOp4default_policy/default_policy/conv_out/kernel/Adam_1Odefault_policy/Placeholder_default_policy/default_policy/conv_out/kernel/Adam_1*
dtype0
?
 default_policy/ReadVariableOp_99ReadVariableOp4default_policy/default_policy/conv_out/kernel/Adam_1#^default_policy/AssignVariableOp_49*'
_output_shapes
:?*
dtype0
?
!default_policy/ReadVariableOp_100ReadVariableOp0default_policy/default_policy/conv_out/bias/Adam*
_output_shapes
:*
dtype0
?
Kdefault_policy/Placeholder_default_policy/default_policy/conv_out/bias/AdamPlaceholder*
_output_shapes
:*
dtype0*
shape:
?
"default_policy/AssignVariableOp_50AssignVariableOp0default_policy/default_policy/conv_out/bias/AdamKdefault_policy/Placeholder_default_policy/default_policy/conv_out/bias/Adam*
dtype0
?
!default_policy/ReadVariableOp_101ReadVariableOp0default_policy/default_policy/conv_out/bias/Adam#^default_policy/AssignVariableOp_50*
_output_shapes
:*
dtype0
?
!default_policy/ReadVariableOp_102ReadVariableOp2default_policy/default_policy/conv_out/bias/Adam_1*
_output_shapes
:*
dtype0
?
Mdefault_policy/Placeholder_default_policy/default_policy/conv_out/bias/Adam_1Placeholder*
_output_shapes
:*
dtype0*
shape:
?
"default_policy/AssignVariableOp_51AssignVariableOp2default_policy/default_policy/conv_out/bias/Adam_1Mdefault_policy/Placeholder_default_policy/default_policy/conv_out/bias/Adam_1*
dtype0
?
!default_policy/ReadVariableOp_103ReadVariableOp2default_policy/default_policy/conv_out/bias/Adam_1#^default_policy/AssignVariableOp_51*
_output_shapes
:*
dtype0
?
default_policy/init_2NoOp"^default_policy/beta1_power/Assign"^default_policy/beta2_power/Assign!^default_policy/conv1/bias/Assign#^default_policy/conv1/kernel/Assign!^default_policy/conv2/bias/Assign#^default_policy/conv2/kernel/Assign!^default_policy/conv3/bias/Assign#^default_policy/conv3/kernel/Assign$^default_policy/conv_out/bias/Assign&^default_policy/conv_out/kernel/Assign(^default_policy/conv_value_1/bias/Assign*^default_policy/conv_value_1/kernel/Assign(^default_policy/conv_value_2/bias/Assign*^default_policy/conv_value_2/kernel/Assign(^default_policy/conv_value_3/bias/Assign*^default_policy/conv_value_3/kernel/Assign*^default_policy/conv_value_out/bias/Assign,^default_policy/conv_value_out/kernel/Assign5^default_policy/default_policy/conv1/bias/Adam/Assign7^default_policy/default_policy/conv1/bias/Adam_1/Assign7^default_policy/default_policy/conv1/kernel/Adam/Assign9^default_policy/default_policy/conv1/kernel/Adam_1/Assign5^default_policy/default_policy/conv2/bias/Adam/Assign7^default_policy/default_policy/conv2/bias/Adam_1/Assign7^default_policy/default_policy/conv2/kernel/Adam/Assign9^default_policy/default_policy/conv2/kernel/Adam_1/Assign5^default_policy/default_policy/conv3/bias/Adam/Assign7^default_policy/default_policy/conv3/bias/Adam_1/Assign7^default_policy/default_policy/conv3/kernel/Adam/Assign9^default_policy/default_policy/conv3/kernel/Adam_1/Assign8^default_policy/default_policy/conv_out/bias/Adam/Assign:^default_policy/default_policy/conv_out/bias/Adam_1/Assign:^default_policy/default_policy/conv_out/kernel/Adam/Assign<^default_policy/default_policy/conv_out/kernel/Adam_1/Assign<^default_policy/default_policy/conv_value_1/bias/Adam/Assign>^default_policy/default_policy/conv_value_1/bias/Adam_1/Assign>^default_policy/default_policy/conv_value_1/kernel/Adam/Assign@^default_policy/default_policy/conv_value_1/kernel/Adam_1/Assign<^default_policy/default_policy/conv_value_2/bias/Adam/Assign>^default_policy/default_policy/conv_value_2/bias/Adam_1/Assign>^default_policy/default_policy/conv_value_2/kernel/Adam/Assign@^default_policy/default_policy/conv_value_2/kernel/Adam_1/Assign<^default_policy/default_policy/conv_value_3/bias/Adam/Assign>^default_policy/default_policy/conv_value_3/bias/Adam_1/Assign>^default_policy/default_policy/conv_value_3/kernel/Adam/Assign@^default_policy/default_policy/conv_value_3/kernel/Adam_1/Assign>^default_policy/default_policy/conv_value_out/bias/Adam/Assign@^default_policy/default_policy/conv_value_out/bias/Adam_1/Assign@^default_policy/default_policy/conv_value_out/kernel/Adam/AssignB^default_policy/default_policy/conv_value_out/kernel/Adam_1/Assign$^default_policy/entropy_coeff/Assign"^default_policy/global_step/Assign^default_policy/kl_coeff/Assign^default_policy/lr/Assign!^default_policy/timestep_1/Assign
Y
save/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
dtype0*
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
dtype0*
shape: 
{
save/StaticRegexFullMatchStaticRegexFullMatch
save/Const"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
a
save/Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
f
save/Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
|
save/SelectSelectsave/StaticRegexFullMatchsave/Const_1save/Const_2"/device:CPU:**
T0*
_output_shapes
: 
w
save/StringJoin
StringJoin
save/Constsave/Select"/device:CPU:**
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
k
save/ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
?
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
?
save/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*?
value?B?7Bdefault_policy/beta1_powerBdefault_policy/beta2_powerBdefault_policy/conv1/biasBdefault_policy/conv1/kernelBdefault_policy/conv2/biasBdefault_policy/conv2/kernelBdefault_policy/conv3/biasBdefault_policy/conv3/kernelBdefault_policy/conv_out/biasBdefault_policy/conv_out/kernelB default_policy/conv_value_1/biasB"default_policy/conv_value_1/kernelB default_policy/conv_value_2/biasB"default_policy/conv_value_2/kernelB default_policy/conv_value_3/biasB"default_policy/conv_value_3/kernelB"default_policy/conv_value_out/biasB$default_policy/conv_value_out/kernelB-default_policy/default_policy/conv1/bias/AdamB/default_policy/default_policy/conv1/bias/Adam_1B/default_policy/default_policy/conv1/kernel/AdamB1default_policy/default_policy/conv1/kernel/Adam_1B-default_policy/default_policy/conv2/bias/AdamB/default_policy/default_policy/conv2/bias/Adam_1B/default_policy/default_policy/conv2/kernel/AdamB1default_policy/default_policy/conv2/kernel/Adam_1B-default_policy/default_policy/conv3/bias/AdamB/default_policy/default_policy/conv3/bias/Adam_1B/default_policy/default_policy/conv3/kernel/AdamB1default_policy/default_policy/conv3/kernel/Adam_1B0default_policy/default_policy/conv_out/bias/AdamB2default_policy/default_policy/conv_out/bias/Adam_1B2default_policy/default_policy/conv_out/kernel/AdamB4default_policy/default_policy/conv_out/kernel/Adam_1B4default_policy/default_policy/conv_value_1/bias/AdamB6default_policy/default_policy/conv_value_1/bias/Adam_1B6default_policy/default_policy/conv_value_1/kernel/AdamB8default_policy/default_policy/conv_value_1/kernel/Adam_1B4default_policy/default_policy/conv_value_2/bias/AdamB6default_policy/default_policy/conv_value_2/bias/Adam_1B6default_policy/default_policy/conv_value_2/kernel/AdamB8default_policy/default_policy/conv_value_2/kernel/Adam_1B4default_policy/default_policy/conv_value_3/bias/AdamB6default_policy/default_policy/conv_value_3/bias/Adam_1B6default_policy/default_policy/conv_value_3/kernel/AdamB8default_policy/default_policy/conv_value_3/kernel/Adam_1B6default_policy/default_policy/conv_value_out/bias/AdamB8default_policy/default_policy/conv_value_out/bias/Adam_1B8default_policy/default_policy/conv_value_out/kernel/AdamB:default_policy/default_policy/conv_value_out/kernel/Adam_1Bdefault_policy/entropy_coeffBdefault_policy/global_stepBdefault_policy/kl_coeffBdefault_policy/lrBdefault_policy/timestep_1
?
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*?
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices.default_policy/beta1_power/Read/ReadVariableOp.default_policy/beta2_power/Read/ReadVariableOp-default_policy/conv1/bias/Read/ReadVariableOp/default_policy/conv1/kernel/Read/ReadVariableOp-default_policy/conv2/bias/Read/ReadVariableOp/default_policy/conv2/kernel/Read/ReadVariableOp-default_policy/conv3/bias/Read/ReadVariableOp/default_policy/conv3/kernel/Read/ReadVariableOp0default_policy/conv_out/bias/Read/ReadVariableOp2default_policy/conv_out/kernel/Read/ReadVariableOp4default_policy/conv_value_1/bias/Read/ReadVariableOp6default_policy/conv_value_1/kernel/Read/ReadVariableOp4default_policy/conv_value_2/bias/Read/ReadVariableOp6default_policy/conv_value_2/kernel/Read/ReadVariableOp4default_policy/conv_value_3/bias/Read/ReadVariableOp6default_policy/conv_value_3/kernel/Read/ReadVariableOp6default_policy/conv_value_out/bias/Read/ReadVariableOp8default_policy/conv_value_out/kernel/Read/ReadVariableOpAdefault_policy/default_policy/conv1/bias/Adam/Read/ReadVariableOpCdefault_policy/default_policy/conv1/bias/Adam_1/Read/ReadVariableOpCdefault_policy/default_policy/conv1/kernel/Adam/Read/ReadVariableOpEdefault_policy/default_policy/conv1/kernel/Adam_1/Read/ReadVariableOpAdefault_policy/default_policy/conv2/bias/Adam/Read/ReadVariableOpCdefault_policy/default_policy/conv2/bias/Adam_1/Read/ReadVariableOpCdefault_policy/default_policy/conv2/kernel/Adam/Read/ReadVariableOpEdefault_policy/default_policy/conv2/kernel/Adam_1/Read/ReadVariableOpAdefault_policy/default_policy/conv3/bias/Adam/Read/ReadVariableOpCdefault_policy/default_policy/conv3/bias/Adam_1/Read/ReadVariableOpCdefault_policy/default_policy/conv3/kernel/Adam/Read/ReadVariableOpEdefault_policy/default_policy/conv3/kernel/Adam_1/Read/ReadVariableOpDdefault_policy/default_policy/conv_out/bias/Adam/Read/ReadVariableOpFdefault_policy/default_policy/conv_out/bias/Adam_1/Read/ReadVariableOpFdefault_policy/default_policy/conv_out/kernel/Adam/Read/ReadVariableOpHdefault_policy/default_policy/conv_out/kernel/Adam_1/Read/ReadVariableOpHdefault_policy/default_policy/conv_value_1/bias/Adam/Read/ReadVariableOpJdefault_policy/default_policy/conv_value_1/bias/Adam_1/Read/ReadVariableOpJdefault_policy/default_policy/conv_value_1/kernel/Adam/Read/ReadVariableOpLdefault_policy/default_policy/conv_value_1/kernel/Adam_1/Read/ReadVariableOpHdefault_policy/default_policy/conv_value_2/bias/Adam/Read/ReadVariableOpJdefault_policy/default_policy/conv_value_2/bias/Adam_1/Read/ReadVariableOpJdefault_policy/default_policy/conv_value_2/kernel/Adam/Read/ReadVariableOpLdefault_policy/default_policy/conv_value_2/kernel/Adam_1/Read/ReadVariableOpHdefault_policy/default_policy/conv_value_3/bias/Adam/Read/ReadVariableOpJdefault_policy/default_policy/conv_value_3/bias/Adam_1/Read/ReadVariableOpJdefault_policy/default_policy/conv_value_3/kernel/Adam/Read/ReadVariableOpLdefault_policy/default_policy/conv_value_3/kernel/Adam_1/Read/ReadVariableOpJdefault_policy/default_policy/conv_value_out/bias/Adam/Read/ReadVariableOpLdefault_policy/default_policy/conv_value_out/bias/Adam_1/Read/ReadVariableOpLdefault_policy/default_policy/conv_value_out/kernel/Adam/Read/ReadVariableOpNdefault_policy/default_policy/conv_value_out/kernel/Adam_1/Read/ReadVariableOp0default_policy/entropy_coeff/Read/ReadVariableOp.default_policy/global_step/Read/ReadVariableOp+default_policy/kl_coeff/Read/ReadVariableOp%default_policy/lr/Read/ReadVariableOp-default_policy/timestep_1/Read/ReadVariableOp"/device:CPU:0*E
dtypes;
927		
?
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
?
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*
T0*
_output_shapes
:*

axis 
?
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(
?
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
?
save/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*?
value?B?7Bdefault_policy/beta1_powerBdefault_policy/beta2_powerBdefault_policy/conv1/biasBdefault_policy/conv1/kernelBdefault_policy/conv2/biasBdefault_policy/conv2/kernelBdefault_policy/conv3/biasBdefault_policy/conv3/kernelBdefault_policy/conv_out/biasBdefault_policy/conv_out/kernelB default_policy/conv_value_1/biasB"default_policy/conv_value_1/kernelB default_policy/conv_value_2/biasB"default_policy/conv_value_2/kernelB default_policy/conv_value_3/biasB"default_policy/conv_value_3/kernelB"default_policy/conv_value_out/biasB$default_policy/conv_value_out/kernelB-default_policy/default_policy/conv1/bias/AdamB/default_policy/default_policy/conv1/bias/Adam_1B/default_policy/default_policy/conv1/kernel/AdamB1default_policy/default_policy/conv1/kernel/Adam_1B-default_policy/default_policy/conv2/bias/AdamB/default_policy/default_policy/conv2/bias/Adam_1B/default_policy/default_policy/conv2/kernel/AdamB1default_policy/default_policy/conv2/kernel/Adam_1B-default_policy/default_policy/conv3/bias/AdamB/default_policy/default_policy/conv3/bias/Adam_1B/default_policy/default_policy/conv3/kernel/AdamB1default_policy/default_policy/conv3/kernel/Adam_1B0default_policy/default_policy/conv_out/bias/AdamB2default_policy/default_policy/conv_out/bias/Adam_1B2default_policy/default_policy/conv_out/kernel/AdamB4default_policy/default_policy/conv_out/kernel/Adam_1B4default_policy/default_policy/conv_value_1/bias/AdamB6default_policy/default_policy/conv_value_1/bias/Adam_1B6default_policy/default_policy/conv_value_1/kernel/AdamB8default_policy/default_policy/conv_value_1/kernel/Adam_1B4default_policy/default_policy/conv_value_2/bias/AdamB6default_policy/default_policy/conv_value_2/bias/Adam_1B6default_policy/default_policy/conv_value_2/kernel/AdamB8default_policy/default_policy/conv_value_2/kernel/Adam_1B4default_policy/default_policy/conv_value_3/bias/AdamB6default_policy/default_policy/conv_value_3/bias/Adam_1B6default_policy/default_policy/conv_value_3/kernel/AdamB8default_policy/default_policy/conv_value_3/kernel/Adam_1B6default_policy/default_policy/conv_value_out/bias/AdamB8default_policy/default_policy/conv_value_out/bias/Adam_1B8default_policy/default_policy/conv_value_out/kernel/AdamB:default_policy/default_policy/conv_value_out/kernel/Adam_1Bdefault_policy/entropy_coeffBdefault_policy/global_stepBdefault_policy/kl_coeffBdefault_policy/lrBdefault_policy/timestep_1
?
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*?
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::*E
dtypes;
927		
N
save/Identity_1Identitysave/RestoreV2*
T0*
_output_shapes
:
c
save/AssignVariableOpAssignVariableOpdefault_policy/beta1_powersave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:1*
T0*
_output_shapes
:
e
save/AssignVariableOp_1AssignVariableOpdefault_policy/beta2_powersave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:2*
T0*
_output_shapes
:
d
save/AssignVariableOp_2AssignVariableOpdefault_policy/conv1/biassave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:3*
T0*
_output_shapes
:
f
save/AssignVariableOp_3AssignVariableOpdefault_policy/conv1/kernelsave/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:4*
T0*
_output_shapes
:
d
save/AssignVariableOp_4AssignVariableOpdefault_policy/conv2/biassave/Identity_5*
dtype0
P
save/Identity_6Identitysave/RestoreV2:5*
T0*
_output_shapes
:
f
save/AssignVariableOp_5AssignVariableOpdefault_policy/conv2/kernelsave/Identity_6*
dtype0
P
save/Identity_7Identitysave/RestoreV2:6*
T0*
_output_shapes
:
d
save/AssignVariableOp_6AssignVariableOpdefault_policy/conv3/biassave/Identity_7*
dtype0
P
save/Identity_8Identitysave/RestoreV2:7*
T0*
_output_shapes
:
f
save/AssignVariableOp_7AssignVariableOpdefault_policy/conv3/kernelsave/Identity_8*
dtype0
P
save/Identity_9Identitysave/RestoreV2:8*
T0*
_output_shapes
:
g
save/AssignVariableOp_8AssignVariableOpdefault_policy/conv_out/biassave/Identity_9*
dtype0
Q
save/Identity_10Identitysave/RestoreV2:9*
T0*
_output_shapes
:
j
save/AssignVariableOp_9AssignVariableOpdefault_policy/conv_out/kernelsave/Identity_10*
dtype0
R
save/Identity_11Identitysave/RestoreV2:10*
T0*
_output_shapes
:
m
save/AssignVariableOp_10AssignVariableOp default_policy/conv_value_1/biassave/Identity_11*
dtype0
R
save/Identity_12Identitysave/RestoreV2:11*
T0*
_output_shapes
:
o
save/AssignVariableOp_11AssignVariableOp"default_policy/conv_value_1/kernelsave/Identity_12*
dtype0
R
save/Identity_13Identitysave/RestoreV2:12*
T0*
_output_shapes
:
m
save/AssignVariableOp_12AssignVariableOp default_policy/conv_value_2/biassave/Identity_13*
dtype0
R
save/Identity_14Identitysave/RestoreV2:13*
T0*
_output_shapes
:
o
save/AssignVariableOp_13AssignVariableOp"default_policy/conv_value_2/kernelsave/Identity_14*
dtype0
R
save/Identity_15Identitysave/RestoreV2:14*
T0*
_output_shapes
:
m
save/AssignVariableOp_14AssignVariableOp default_policy/conv_value_3/biassave/Identity_15*
dtype0
R
save/Identity_16Identitysave/RestoreV2:15*
T0*
_output_shapes
:
o
save/AssignVariableOp_15AssignVariableOp"default_policy/conv_value_3/kernelsave/Identity_16*
dtype0
R
save/Identity_17Identitysave/RestoreV2:16*
T0*
_output_shapes
:
o
save/AssignVariableOp_16AssignVariableOp"default_policy/conv_value_out/biassave/Identity_17*
dtype0
R
save/Identity_18Identitysave/RestoreV2:17*
T0*
_output_shapes
:
q
save/AssignVariableOp_17AssignVariableOp$default_policy/conv_value_out/kernelsave/Identity_18*
dtype0
R
save/Identity_19Identitysave/RestoreV2:18*
T0*
_output_shapes
:
z
save/AssignVariableOp_18AssignVariableOp-default_policy/default_policy/conv1/bias/Adamsave/Identity_19*
dtype0
R
save/Identity_20Identitysave/RestoreV2:19*
T0*
_output_shapes
:
|
save/AssignVariableOp_19AssignVariableOp/default_policy/default_policy/conv1/bias/Adam_1save/Identity_20*
dtype0
R
save/Identity_21Identitysave/RestoreV2:20*
T0*
_output_shapes
:
|
save/AssignVariableOp_20AssignVariableOp/default_policy/default_policy/conv1/kernel/Adamsave/Identity_21*
dtype0
R
save/Identity_22Identitysave/RestoreV2:21*
T0*
_output_shapes
:
~
save/AssignVariableOp_21AssignVariableOp1default_policy/default_policy/conv1/kernel/Adam_1save/Identity_22*
dtype0
R
save/Identity_23Identitysave/RestoreV2:22*
T0*
_output_shapes
:
z
save/AssignVariableOp_22AssignVariableOp-default_policy/default_policy/conv2/bias/Adamsave/Identity_23*
dtype0
R
save/Identity_24Identitysave/RestoreV2:23*
T0*
_output_shapes
:
|
save/AssignVariableOp_23AssignVariableOp/default_policy/default_policy/conv2/bias/Adam_1save/Identity_24*
dtype0
R
save/Identity_25Identitysave/RestoreV2:24*
T0*
_output_shapes
:
|
save/AssignVariableOp_24AssignVariableOp/default_policy/default_policy/conv2/kernel/Adamsave/Identity_25*
dtype0
R
save/Identity_26Identitysave/RestoreV2:25*
T0*
_output_shapes
:
~
save/AssignVariableOp_25AssignVariableOp1default_policy/default_policy/conv2/kernel/Adam_1save/Identity_26*
dtype0
R
save/Identity_27Identitysave/RestoreV2:26*
T0*
_output_shapes
:
z
save/AssignVariableOp_26AssignVariableOp-default_policy/default_policy/conv3/bias/Adamsave/Identity_27*
dtype0
R
save/Identity_28Identitysave/RestoreV2:27*
T0*
_output_shapes
:
|
save/AssignVariableOp_27AssignVariableOp/default_policy/default_policy/conv3/bias/Adam_1save/Identity_28*
dtype0
R
save/Identity_29Identitysave/RestoreV2:28*
T0*
_output_shapes
:
|
save/AssignVariableOp_28AssignVariableOp/default_policy/default_policy/conv3/kernel/Adamsave/Identity_29*
dtype0
R
save/Identity_30Identitysave/RestoreV2:29*
T0*
_output_shapes
:
~
save/AssignVariableOp_29AssignVariableOp1default_policy/default_policy/conv3/kernel/Adam_1save/Identity_30*
dtype0
R
save/Identity_31Identitysave/RestoreV2:30*
T0*
_output_shapes
:
}
save/AssignVariableOp_30AssignVariableOp0default_policy/default_policy/conv_out/bias/Adamsave/Identity_31*
dtype0
R
save/Identity_32Identitysave/RestoreV2:31*
T0*
_output_shapes
:

save/AssignVariableOp_31AssignVariableOp2default_policy/default_policy/conv_out/bias/Adam_1save/Identity_32*
dtype0
R
save/Identity_33Identitysave/RestoreV2:32*
T0*
_output_shapes
:

save/AssignVariableOp_32AssignVariableOp2default_policy/default_policy/conv_out/kernel/Adamsave/Identity_33*
dtype0
R
save/Identity_34Identitysave/RestoreV2:33*
T0*
_output_shapes
:
?
save/AssignVariableOp_33AssignVariableOp4default_policy/default_policy/conv_out/kernel/Adam_1save/Identity_34*
dtype0
R
save/Identity_35Identitysave/RestoreV2:34*
T0*
_output_shapes
:
?
save/AssignVariableOp_34AssignVariableOp4default_policy/default_policy/conv_value_1/bias/Adamsave/Identity_35*
dtype0
R
save/Identity_36Identitysave/RestoreV2:35*
T0*
_output_shapes
:
?
save/AssignVariableOp_35AssignVariableOp6default_policy/default_policy/conv_value_1/bias/Adam_1save/Identity_36*
dtype0
R
save/Identity_37Identitysave/RestoreV2:36*
T0*
_output_shapes
:
?
save/AssignVariableOp_36AssignVariableOp6default_policy/default_policy/conv_value_1/kernel/Adamsave/Identity_37*
dtype0
R
save/Identity_38Identitysave/RestoreV2:37*
T0*
_output_shapes
:
?
save/AssignVariableOp_37AssignVariableOp8default_policy/default_policy/conv_value_1/kernel/Adam_1save/Identity_38*
dtype0
R
save/Identity_39Identitysave/RestoreV2:38*
T0*
_output_shapes
:
?
save/AssignVariableOp_38AssignVariableOp4default_policy/default_policy/conv_value_2/bias/Adamsave/Identity_39*
dtype0
R
save/Identity_40Identitysave/RestoreV2:39*
T0*
_output_shapes
:
?
save/AssignVariableOp_39AssignVariableOp6default_policy/default_policy/conv_value_2/bias/Adam_1save/Identity_40*
dtype0
R
save/Identity_41Identitysave/RestoreV2:40*
T0*
_output_shapes
:
?
save/AssignVariableOp_40AssignVariableOp6default_policy/default_policy/conv_value_2/kernel/Adamsave/Identity_41*
dtype0
R
save/Identity_42Identitysave/RestoreV2:41*
T0*
_output_shapes
:
?
save/AssignVariableOp_41AssignVariableOp8default_policy/default_policy/conv_value_2/kernel/Adam_1save/Identity_42*
dtype0
R
save/Identity_43Identitysave/RestoreV2:42*
T0*
_output_shapes
:
?
save/AssignVariableOp_42AssignVariableOp4default_policy/default_policy/conv_value_3/bias/Adamsave/Identity_43*
dtype0
R
save/Identity_44Identitysave/RestoreV2:43*
T0*
_output_shapes
:
?
save/AssignVariableOp_43AssignVariableOp6default_policy/default_policy/conv_value_3/bias/Adam_1save/Identity_44*
dtype0
R
save/Identity_45Identitysave/RestoreV2:44*
T0*
_output_shapes
:
?
save/AssignVariableOp_44AssignVariableOp6default_policy/default_policy/conv_value_3/kernel/Adamsave/Identity_45*
dtype0
R
save/Identity_46Identitysave/RestoreV2:45*
T0*
_output_shapes
:
?
save/AssignVariableOp_45AssignVariableOp8default_policy/default_policy/conv_value_3/kernel/Adam_1save/Identity_46*
dtype0
R
save/Identity_47Identitysave/RestoreV2:46*
T0*
_output_shapes
:
?
save/AssignVariableOp_46AssignVariableOp6default_policy/default_policy/conv_value_out/bias/Adamsave/Identity_47*
dtype0
R
save/Identity_48Identitysave/RestoreV2:47*
T0*
_output_shapes
:
?
save/AssignVariableOp_47AssignVariableOp8default_policy/default_policy/conv_value_out/bias/Adam_1save/Identity_48*
dtype0
R
save/Identity_49Identitysave/RestoreV2:48*
T0*
_output_shapes
:
?
save/AssignVariableOp_48AssignVariableOp8default_policy/default_policy/conv_value_out/kernel/Adamsave/Identity_49*
dtype0
R
save/Identity_50Identitysave/RestoreV2:49*
T0*
_output_shapes
:
?
save/AssignVariableOp_49AssignVariableOp:default_policy/default_policy/conv_value_out/kernel/Adam_1save/Identity_50*
dtype0
R
save/Identity_51Identitysave/RestoreV2:50*
T0*
_output_shapes
:
i
save/AssignVariableOp_50AssignVariableOpdefault_policy/entropy_coeffsave/Identity_51*
dtype0
R
save/Identity_52Identitysave/RestoreV2:51*
T0	*
_output_shapes
:
g
save/AssignVariableOp_51AssignVariableOpdefault_policy/global_stepsave/Identity_52*
dtype0	
R
save/Identity_53Identitysave/RestoreV2:52*
T0*
_output_shapes
:
d
save/AssignVariableOp_52AssignVariableOpdefault_policy/kl_coeffsave/Identity_53*
dtype0
R
save/Identity_54Identitysave/RestoreV2:53*
T0*
_output_shapes
:
^
save/AssignVariableOp_53AssignVariableOpdefault_policy/lrsave/Identity_54*
dtype0
R
save/Identity_55Identitysave/RestoreV2:54*
T0	*
_output_shapes
:
f
save/AssignVariableOp_54AssignVariableOpdefault_policy/timestep_1save/Identity_55*
dtype0	
?
save/restore_shardNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_12^save/AssignVariableOp_13^save/AssignVariableOp_14^save/AssignVariableOp_15^save/AssignVariableOp_16^save/AssignVariableOp_17^save/AssignVariableOp_18^save/AssignVariableOp_19^save/AssignVariableOp_2^save/AssignVariableOp_20^save/AssignVariableOp_21^save/AssignVariableOp_22^save/AssignVariableOp_23^save/AssignVariableOp_24^save/AssignVariableOp_25^save/AssignVariableOp_26^save/AssignVariableOp_27^save/AssignVariableOp_28^save/AssignVariableOp_29^save/AssignVariableOp_3^save/AssignVariableOp_30^save/AssignVariableOp_31^save/AssignVariableOp_32^save/AssignVariableOp_33^save/AssignVariableOp_34^save/AssignVariableOp_35^save/AssignVariableOp_36^save/AssignVariableOp_37^save/AssignVariableOp_38^save/AssignVariableOp_39^save/AssignVariableOp_4^save/AssignVariableOp_40^save/AssignVariableOp_41^save/AssignVariableOp_42^save/AssignVariableOp_43^save/AssignVariableOp_44^save/AssignVariableOp_45^save/AssignVariableOp_46^save/AssignVariableOp_47^save/AssignVariableOp_48^save/AssignVariableOp_49^save/AssignVariableOp_5^save/AssignVariableOp_50^save/AssignVariableOp_51^save/AssignVariableOp_52^save/AssignVariableOp_53^save/AssignVariableOp_54^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9
-
save/restore_allNoOp^save/restore_shard"?<
save/Const:0save/Identity:0save/restore_all (5 @F8"?'
cond_context?&?&
?
default_policy/cond/cond_textdefault_policy/cond/pred_id:0default_policy/cond/switch_t:0 *?	
default_policy/cond/Const:0
(default_policy/cond/cond/ArgMax/Switch:0
+default_policy/cond/cond/ArgMax/dimension:0
!default_policy/cond/cond/ArgMax:0
 default_policy/cond/cond/Merge:0
 default_policy/cond/cond/Merge:1
'default_policy/cond/cond/Shape/Switch:1
)default_policy/cond/cond/Shape/Switch_1:1
 default_policy/cond/cond/Shape:0
!default_policy/cond/cond/Switch:0
!default_policy/cond/cond/Switch:1
"default_policy/cond/cond/pred_id:0
-default_policy/cond/cond/random_uniform/max:0
-default_policy/cond/cond/random_uniform/min:0
/default_policy/cond/cond/random_uniform/shape:0
)default_policy/cond/cond/random_uniform:0
.default_policy/cond/cond/strided_slice/stack:0
0default_policy/cond/cond/strided_slice/stack_1:0
0default_policy/cond/cond/strided_slice/stack_2:0
(default_policy/cond/cond/strided_slice:0
#default_policy/cond/cond/switch_f:0
#default_policy/cond/cond/switch_t:0
default_policy/cond/pred_id:0
default_policy/cond/switch_t:0
&default_policy/cond/zeros_like/Const:0
&default_policy/cond/zeros_like/Shape:0
 default_policy/cond/zeros_like:0
default_policy/truediv:0>
default_policy/cond/pred_id:0default_policy/cond/pred_id:0C
default_policy/truediv:0'default_policy/cond/cond/Shape/Switch:12?
?
"default_policy/cond/cond/cond_text"default_policy/cond/cond/pred_id:0#default_policy/cond/cond/switch_t:0 *?
'default_policy/cond/cond/Shape/Switch:1
)default_policy/cond/cond/Shape/Switch_1:1
 default_policy/cond/cond/Shape:0
"default_policy/cond/cond/pred_id:0
-default_policy/cond/cond/random_uniform/max:0
-default_policy/cond/cond/random_uniform/min:0
/default_policy/cond/cond/random_uniform/shape:0
)default_policy/cond/cond/random_uniform:0
.default_policy/cond/cond/strided_slice/stack:0
0default_policy/cond/cond/strided_slice/stack_1:0
0default_policy/cond/cond/strided_slice/stack_2:0
(default_policy/cond/cond/strided_slice:0
#default_policy/cond/cond/switch_t:0
default_policy/truediv:0R
'default_policy/cond/cond/Shape/Switch:1'default_policy/cond/cond/Shape/Switch:1H
"default_policy/cond/cond/pred_id:0"default_policy/cond/cond/pred_id:0E
default_policy/truediv:0)default_policy/cond/cond/Shape/Switch_1:12?
?
$default_policy/cond/cond/cond_text_1"default_policy/cond/cond/pred_id:0#default_policy/cond/cond/switch_f:0*?
(default_policy/cond/cond/ArgMax/Switch:0
+default_policy/cond/cond/ArgMax/dimension:0
!default_policy/cond/cond/ArgMax:0
'default_policy/cond/cond/Shape/Switch:1
"default_policy/cond/cond/pred_id:0
#default_policy/cond/cond/switch_f:0
default_policy/truediv:0D
default_policy/truediv:0(default_policy/cond/cond/ArgMax/Switch:0H
"default_policy/cond/cond/pred_id:0"default_policy/cond/cond/pred_id:0R
'default_policy/cond/cond/Shape/Switch:1'default_policy/cond/cond/Shape/Switch:1
?
default_policy/cond/cond_text_1default_policy/cond/pred_id:0default_policy/cond/switch_f:0*?
default_policy/Squeeze_1:0
default_policy/cond/Switch_1:0
default_policy/cond/Switch_1:1
default_policy/cond/pred_id:0
default_policy/cond/switch_f:0>
default_policy/cond/pred_id:0default_policy/cond/pred_id:0<
default_policy/Squeeze_1:0default_policy/cond/Switch_1:0
?
default_policy/cond_1/cond_textdefault_policy/cond_1/pred_id:0 default_policy/cond_1/switch_t:0 *?
default_policy/cond/Merge:0
 default_policy/cond_1/Switch_1:0
 default_policy/cond_1/Switch_1:1
default_policy/cond_1/pred_id:0
 default_policy/cond_1/switch_t:0B
default_policy/cond_1/pred_id:0default_policy/cond_1/pred_id:0?
default_policy/cond/Merge:0 default_policy/cond_1/Switch_1:1
?
!default_policy/cond_1/cond_text_1default_policy/cond_1/pred_id:0 default_policy/cond_1/switch_f:0*?
default_policy/ArgMax:0
 default_policy/cond_1/Switch_2:0
 default_policy/cond_1/Switch_2:1
default_policy/cond_1/pred_id:0
 default_policy/cond_1/switch_f:0;
default_policy/ArgMax:0 default_policy/cond_1/Switch_2:0B
default_policy/cond_1/pred_id:0default_policy/cond_1/pred_id:0
?
default_policy/cond_2/cond_textdefault_policy/cond_2/pred_id:0 default_policy/cond_2/switch_t:0 *?
default_policy/Neg:0
 default_policy/cond_2/Switch_1:0
 default_policy/cond_2/Switch_1:1
default_policy/cond_2/pred_id:0
 default_policy/cond_2/switch_t:08
default_policy/Neg:0 default_policy/cond_2/Switch_1:1B
default_policy/cond_2/pred_id:0default_policy/cond_2/pred_id:0
?
!default_policy/cond_2/cond_text_1default_policy/cond_2/pred_id:0 default_policy/cond_2/switch_f:0*?
default_policy/ArgMax:0
default_policy/cond_2/pred_id:0
 default_policy/cond_2/switch_f:0
(default_policy/cond_2/zeros_like/Const:0
/default_policy/cond_2/zeros_like/Shape/Switch:0
(default_policy/cond_2/zeros_like/Shape:0
"default_policy/cond_2/zeros_like:0B
default_policy/cond_2/pred_id:0default_policy/cond_2/pred_id:0J
default_policy/ArgMax:0/default_policy/cond_2/zeros_like/Shape/Switch:0"?
global_step??
?
default_policy/global_step:0!default_policy/global_step/Assign0default_policy/global_step/Read/ReadVariableOp:0(2.default_policy/global_step/Initializer/zeros:0H"#
train_op

default_policy/Adam"?
trainable_variables??
?
default_policy/conv1/kernel:0"default_policy/conv1/kernel/Assign1default_policy/conv1/kernel/Read/ReadVariableOp:0(28default_policy/conv1/kernel/Initializer/random_uniform:08
?
default_policy/conv1/bias:0 default_policy/conv1/bias/Assign/default_policy/conv1/bias/Read/ReadVariableOp:0(2-default_policy/conv1/bias/Initializer/zeros:08
?
default_policy/conv2/kernel:0"default_policy/conv2/kernel/Assign1default_policy/conv2/kernel/Read/ReadVariableOp:0(28default_policy/conv2/kernel/Initializer/random_uniform:08
?
default_policy/conv2/bias:0 default_policy/conv2/bias/Assign/default_policy/conv2/bias/Read/ReadVariableOp:0(2-default_policy/conv2/bias/Initializer/zeros:08
?
default_policy/conv3/kernel:0"default_policy/conv3/kernel/Assign1default_policy/conv3/kernel/Read/ReadVariableOp:0(28default_policy/conv3/kernel/Initializer/random_uniform:08
?
default_policy/conv3/bias:0 default_policy/conv3/bias/Assign/default_policy/conv3/bias/Read/ReadVariableOp:0(2-default_policy/conv3/bias/Initializer/zeros:08
?
 default_policy/conv_out/kernel:0%default_policy/conv_out/kernel/Assign4default_policy/conv_out/kernel/Read/ReadVariableOp:0(2;default_policy/conv_out/kernel/Initializer/random_uniform:08
?
default_policy/conv_out/bias:0#default_policy/conv_out/bias/Assign2default_policy/conv_out/bias/Read/ReadVariableOp:0(20default_policy/conv_out/bias/Initializer/zeros:08
?
$default_policy/conv_value_1/kernel:0)default_policy/conv_value_1/kernel/Assign8default_policy/conv_value_1/kernel/Read/ReadVariableOp:0(2?default_policy/conv_value_1/kernel/Initializer/random_uniform:08
?
"default_policy/conv_value_1/bias:0'default_policy/conv_value_1/bias/Assign6default_policy/conv_value_1/bias/Read/ReadVariableOp:0(24default_policy/conv_value_1/bias/Initializer/zeros:08
?
$default_policy/conv_value_2/kernel:0)default_policy/conv_value_2/kernel/Assign8default_policy/conv_value_2/kernel/Read/ReadVariableOp:0(2?default_policy/conv_value_2/kernel/Initializer/random_uniform:08
?
"default_policy/conv_value_2/bias:0'default_policy/conv_value_2/bias/Assign6default_policy/conv_value_2/bias/Read/ReadVariableOp:0(24default_policy/conv_value_2/bias/Initializer/zeros:08
?
$default_policy/conv_value_3/kernel:0)default_policy/conv_value_3/kernel/Assign8default_policy/conv_value_3/kernel/Read/ReadVariableOp:0(2?default_policy/conv_value_3/kernel/Initializer/random_uniform:08
?
"default_policy/conv_value_3/bias:0'default_policy/conv_value_3/bias/Assign6default_policy/conv_value_3/bias/Read/ReadVariableOp:0(24default_policy/conv_value_3/bias/Initializer/zeros:08
?
&default_policy/conv_value_out/kernel:0+default_policy/conv_value_out/kernel/Assign:default_policy/conv_value_out/kernel/Read/ReadVariableOp:0(2Adefault_policy/conv_value_out/kernel/Initializer/random_uniform:08
?
$default_policy/conv_value_out/bias:0)default_policy/conv_value_out/bias/Assign8default_policy/conv_value_out/bias/Read/ReadVariableOp:0(26default_policy/conv_value_out/bias/Initializer/zeros:08"?d
	variables?d?d
?
default_policy/conv1/kernel:0"default_policy/conv1/kernel/Assign1default_policy/conv1/kernel/Read/ReadVariableOp:0(28default_policy/conv1/kernel/Initializer/random_uniform:08
?
default_policy/conv1/bias:0 default_policy/conv1/bias/Assign/default_policy/conv1/bias/Read/ReadVariableOp:0(2-default_policy/conv1/bias/Initializer/zeros:08
?
default_policy/conv2/kernel:0"default_policy/conv2/kernel/Assign1default_policy/conv2/kernel/Read/ReadVariableOp:0(28default_policy/conv2/kernel/Initializer/random_uniform:08
?
default_policy/conv2/bias:0 default_policy/conv2/bias/Assign/default_policy/conv2/bias/Read/ReadVariableOp:0(2-default_policy/conv2/bias/Initializer/zeros:08
?
default_policy/conv3/kernel:0"default_policy/conv3/kernel/Assign1default_policy/conv3/kernel/Read/ReadVariableOp:0(28default_policy/conv3/kernel/Initializer/random_uniform:08
?
default_policy/conv3/bias:0 default_policy/conv3/bias/Assign/default_policy/conv3/bias/Read/ReadVariableOp:0(2-default_policy/conv3/bias/Initializer/zeros:08
?
 default_policy/conv_out/kernel:0%default_policy/conv_out/kernel/Assign4default_policy/conv_out/kernel/Read/ReadVariableOp:0(2;default_policy/conv_out/kernel/Initializer/random_uniform:08
?
default_policy/conv_out/bias:0#default_policy/conv_out/bias/Assign2default_policy/conv_out/bias/Read/ReadVariableOp:0(20default_policy/conv_out/bias/Initializer/zeros:08
?
$default_policy/conv_value_1/kernel:0)default_policy/conv_value_1/kernel/Assign8default_policy/conv_value_1/kernel/Read/ReadVariableOp:0(2?default_policy/conv_value_1/kernel/Initializer/random_uniform:08
?
"default_policy/conv_value_1/bias:0'default_policy/conv_value_1/bias/Assign6default_policy/conv_value_1/bias/Read/ReadVariableOp:0(24default_policy/conv_value_1/bias/Initializer/zeros:08
?
$default_policy/conv_value_2/kernel:0)default_policy/conv_value_2/kernel/Assign8default_policy/conv_value_2/kernel/Read/ReadVariableOp:0(2?default_policy/conv_value_2/kernel/Initializer/random_uniform:08
?
"default_policy/conv_value_2/bias:0'default_policy/conv_value_2/bias/Assign6default_policy/conv_value_2/bias/Read/ReadVariableOp:0(24default_policy/conv_value_2/bias/Initializer/zeros:08
?
$default_policy/conv_value_3/kernel:0)default_policy/conv_value_3/kernel/Assign8default_policy/conv_value_3/kernel/Read/ReadVariableOp:0(2?default_policy/conv_value_3/kernel/Initializer/random_uniform:08
?
"default_policy/conv_value_3/bias:0'default_policy/conv_value_3/bias/Assign6default_policy/conv_value_3/bias/Read/ReadVariableOp:0(24default_policy/conv_value_3/bias/Initializer/zeros:08
?
&default_policy/conv_value_out/kernel:0+default_policy/conv_value_out/kernel/Assign:default_policy/conv_value_out/kernel/Read/ReadVariableOp:0(2Adefault_policy/conv_value_out/kernel/Initializer/random_uniform:08
?
$default_policy/conv_value_out/bias:0)default_policy/conv_value_out/bias/Assign8default_policy/conv_value_out/bias/Read/ReadVariableOp:0(26default_policy/conv_value_out/bias/Initializer/zeros:08
?
default_policy/timestep_1:0 default_policy/timestep_1/Assign/default_policy/timestep_1/Read/ReadVariableOp:0(25default_policy/timestep_1/Initializer/initial_value:0
?
default_policy/kl_coeff:0default_policy/kl_coeff/Assign-default_policy/kl_coeff/Read/ReadVariableOp:0(23default_policy/kl_coeff/Initializer/initial_value:0
?
default_policy/entropy_coeff:0#default_policy/entropy_coeff/Assign2default_policy/entropy_coeff/Read/ReadVariableOp:0(28default_policy/entropy_coeff/Initializer/initial_value:0
?
default_policy/lr:0default_policy/lr/Assign'default_policy/lr/Read/ReadVariableOp:0(2-default_policy/lr/Initializer/initial_value:0
?
default_policy/global_step:0!default_policy/global_step/Assign0default_policy/global_step/Read/ReadVariableOp:0(2.default_policy/global_step/Initializer/zeros:0H
?
default_policy/beta1_power:0!default_policy/beta1_power/Assign0default_policy/beta1_power/Read/ReadVariableOp:0(26default_policy/beta1_power/Initializer/initial_value:0
?
default_policy/beta2_power:0!default_policy/beta2_power/Assign0default_policy/beta2_power/Read/ReadVariableOp:0(26default_policy/beta2_power/Initializer/initial_value:0
?
8default_policy/default_policy/conv_value_1/kernel/Adam:0=default_policy/default_policy/conv_value_1/kernel/Adam/AssignLdefault_policy/default_policy/conv_value_1/kernel/Adam/Read/ReadVariableOp:0(2Jdefault_policy/default_policy/conv_value_1/kernel/Adam/Initializer/zeros:0
?
:default_policy/default_policy/conv_value_1/kernel/Adam_1:0?default_policy/default_policy/conv_value_1/kernel/Adam_1/AssignNdefault_policy/default_policy/conv_value_1/kernel/Adam_1/Read/ReadVariableOp:0(2Ldefault_policy/default_policy/conv_value_1/kernel/Adam_1/Initializer/zeros:0
?
6default_policy/default_policy/conv_value_1/bias/Adam:0;default_policy/default_policy/conv_value_1/bias/Adam/AssignJdefault_policy/default_policy/conv_value_1/bias/Adam/Read/ReadVariableOp:0(2Hdefault_policy/default_policy/conv_value_1/bias/Adam/Initializer/zeros:0
?
8default_policy/default_policy/conv_value_1/bias/Adam_1:0=default_policy/default_policy/conv_value_1/bias/Adam_1/AssignLdefault_policy/default_policy/conv_value_1/bias/Adam_1/Read/ReadVariableOp:0(2Jdefault_policy/default_policy/conv_value_1/bias/Adam_1/Initializer/zeros:0
?
1default_policy/default_policy/conv1/kernel/Adam:06default_policy/default_policy/conv1/kernel/Adam/AssignEdefault_policy/default_policy/conv1/kernel/Adam/Read/ReadVariableOp:0(2Cdefault_policy/default_policy/conv1/kernel/Adam/Initializer/zeros:0
?
3default_policy/default_policy/conv1/kernel/Adam_1:08default_policy/default_policy/conv1/kernel/Adam_1/AssignGdefault_policy/default_policy/conv1/kernel/Adam_1/Read/ReadVariableOp:0(2Edefault_policy/default_policy/conv1/kernel/Adam_1/Initializer/zeros:0
?
/default_policy/default_policy/conv1/bias/Adam:04default_policy/default_policy/conv1/bias/Adam/AssignCdefault_policy/default_policy/conv1/bias/Adam/Read/ReadVariableOp:0(2Adefault_policy/default_policy/conv1/bias/Adam/Initializer/zeros:0
?
1default_policy/default_policy/conv1/bias/Adam_1:06default_policy/default_policy/conv1/bias/Adam_1/AssignEdefault_policy/default_policy/conv1/bias/Adam_1/Read/ReadVariableOp:0(2Cdefault_policy/default_policy/conv1/bias/Adam_1/Initializer/zeros:0
?
8default_policy/default_policy/conv_value_2/kernel/Adam:0=default_policy/default_policy/conv_value_2/kernel/Adam/AssignLdefault_policy/default_policy/conv_value_2/kernel/Adam/Read/ReadVariableOp:0(2Jdefault_policy/default_policy/conv_value_2/kernel/Adam/Initializer/zeros:0
?
:default_policy/default_policy/conv_value_2/kernel/Adam_1:0?default_policy/default_policy/conv_value_2/kernel/Adam_1/AssignNdefault_policy/default_policy/conv_value_2/kernel/Adam_1/Read/ReadVariableOp:0(2Ldefault_policy/default_policy/conv_value_2/kernel/Adam_1/Initializer/zeros:0
?
6default_policy/default_policy/conv_value_2/bias/Adam:0;default_policy/default_policy/conv_value_2/bias/Adam/AssignJdefault_policy/default_policy/conv_value_2/bias/Adam/Read/ReadVariableOp:0(2Hdefault_policy/default_policy/conv_value_2/bias/Adam/Initializer/zeros:0
?
8default_policy/default_policy/conv_value_2/bias/Adam_1:0=default_policy/default_policy/conv_value_2/bias/Adam_1/AssignLdefault_policy/default_policy/conv_value_2/bias/Adam_1/Read/ReadVariableOp:0(2Jdefault_policy/default_policy/conv_value_2/bias/Adam_1/Initializer/zeros:0
?
1default_policy/default_policy/conv2/kernel/Adam:06default_policy/default_policy/conv2/kernel/Adam/AssignEdefault_policy/default_policy/conv2/kernel/Adam/Read/ReadVariableOp:0(2Cdefault_policy/default_policy/conv2/kernel/Adam/Initializer/zeros:0
?
3default_policy/default_policy/conv2/kernel/Adam_1:08default_policy/default_policy/conv2/kernel/Adam_1/AssignGdefault_policy/default_policy/conv2/kernel/Adam_1/Read/ReadVariableOp:0(2Edefault_policy/default_policy/conv2/kernel/Adam_1/Initializer/zeros:0
?
/default_policy/default_policy/conv2/bias/Adam:04default_policy/default_policy/conv2/bias/Adam/AssignCdefault_policy/default_policy/conv2/bias/Adam/Read/ReadVariableOp:0(2Adefault_policy/default_policy/conv2/bias/Adam/Initializer/zeros:0
?
1default_policy/default_policy/conv2/bias/Adam_1:06default_policy/default_policy/conv2/bias/Adam_1/AssignEdefault_policy/default_policy/conv2/bias/Adam_1/Read/ReadVariableOp:0(2Cdefault_policy/default_policy/conv2/bias/Adam_1/Initializer/zeros:0
?
8default_policy/default_policy/conv_value_3/kernel/Adam:0=default_policy/default_policy/conv_value_3/kernel/Adam/AssignLdefault_policy/default_policy/conv_value_3/kernel/Adam/Read/ReadVariableOp:0(2Jdefault_policy/default_policy/conv_value_3/kernel/Adam/Initializer/zeros:0
?
:default_policy/default_policy/conv_value_3/kernel/Adam_1:0?default_policy/default_policy/conv_value_3/kernel/Adam_1/AssignNdefault_policy/default_policy/conv_value_3/kernel/Adam_1/Read/ReadVariableOp:0(2Ldefault_policy/default_policy/conv_value_3/kernel/Adam_1/Initializer/zeros:0
?
6default_policy/default_policy/conv_value_3/bias/Adam:0;default_policy/default_policy/conv_value_3/bias/Adam/AssignJdefault_policy/default_policy/conv_value_3/bias/Adam/Read/ReadVariableOp:0(2Hdefault_policy/default_policy/conv_value_3/bias/Adam/Initializer/zeros:0
?
8default_policy/default_policy/conv_value_3/bias/Adam_1:0=default_policy/default_policy/conv_value_3/bias/Adam_1/AssignLdefault_policy/default_policy/conv_value_3/bias/Adam_1/Read/ReadVariableOp:0(2Jdefault_policy/default_policy/conv_value_3/bias/Adam_1/Initializer/zeros:0
?
1default_policy/default_policy/conv3/kernel/Adam:06default_policy/default_policy/conv3/kernel/Adam/AssignEdefault_policy/default_policy/conv3/kernel/Adam/Read/ReadVariableOp:0(2Cdefault_policy/default_policy/conv3/kernel/Adam/Initializer/zeros:0
?
3default_policy/default_policy/conv3/kernel/Adam_1:08default_policy/default_policy/conv3/kernel/Adam_1/AssignGdefault_policy/default_policy/conv3/kernel/Adam_1/Read/ReadVariableOp:0(2Edefault_policy/default_policy/conv3/kernel/Adam_1/Initializer/zeros:0
?
/default_policy/default_policy/conv3/bias/Adam:04default_policy/default_policy/conv3/bias/Adam/AssignCdefault_policy/default_policy/conv3/bias/Adam/Read/ReadVariableOp:0(2Adefault_policy/default_policy/conv3/bias/Adam/Initializer/zeros:0
?
1default_policy/default_policy/conv3/bias/Adam_1:06default_policy/default_policy/conv3/bias/Adam_1/AssignEdefault_policy/default_policy/conv3/bias/Adam_1/Read/ReadVariableOp:0(2Cdefault_policy/default_policy/conv3/bias/Adam_1/Initializer/zeros:0
?
:default_policy/default_policy/conv_value_out/kernel/Adam:0?default_policy/default_policy/conv_value_out/kernel/Adam/AssignNdefault_policy/default_policy/conv_value_out/kernel/Adam/Read/ReadVariableOp:0(2Ldefault_policy/default_policy/conv_value_out/kernel/Adam/Initializer/zeros:0
?
<default_policy/default_policy/conv_value_out/kernel/Adam_1:0Adefault_policy/default_policy/conv_value_out/kernel/Adam_1/AssignPdefault_policy/default_policy/conv_value_out/kernel/Adam_1/Read/ReadVariableOp:0(2Ndefault_policy/default_policy/conv_value_out/kernel/Adam_1/Initializer/zeros:0
?
8default_policy/default_policy/conv_value_out/bias/Adam:0=default_policy/default_policy/conv_value_out/bias/Adam/AssignLdefault_policy/default_policy/conv_value_out/bias/Adam/Read/ReadVariableOp:0(2Jdefault_policy/default_policy/conv_value_out/bias/Adam/Initializer/zeros:0
?
:default_policy/default_policy/conv_value_out/bias/Adam_1:0?default_policy/default_policy/conv_value_out/bias/Adam_1/AssignNdefault_policy/default_policy/conv_value_out/bias/Adam_1/Read/ReadVariableOp:0(2Ldefault_policy/default_policy/conv_value_out/bias/Adam_1/Initializer/zeros:0
?
4default_policy/default_policy/conv_out/kernel/Adam:09default_policy/default_policy/conv_out/kernel/Adam/AssignHdefault_policy/default_policy/conv_out/kernel/Adam/Read/ReadVariableOp:0(2Fdefault_policy/default_policy/conv_out/kernel/Adam/Initializer/zeros:0
?
6default_policy/default_policy/conv_out/kernel/Adam_1:0;default_policy/default_policy/conv_out/kernel/Adam_1/AssignJdefault_policy/default_policy/conv_out/kernel/Adam_1/Read/ReadVariableOp:0(2Hdefault_policy/default_policy/conv_out/kernel/Adam_1/Initializer/zeros:0
?
2default_policy/default_policy/conv_out/bias/Adam:07default_policy/default_policy/conv_out/bias/Adam/AssignFdefault_policy/default_policy/conv_out/bias/Adam/Read/ReadVariableOp:0(2Ddefault_policy/default_policy/conv_out/bias/Adam/Initializer/zeros:0
?
4default_policy/default_policy/conv_out/bias/Adam_1:09default_policy/default_policy/conv_out/bias/Adam_1/AssignHdefault_policy/default_policy/conv_out/bias/Adam_1/Read/ReadVariableOp:0(2Fdefault_policy/default_policy/conv_out/bias/Adam_1/Initializer/zeros:0*?
serving_default?
1
is_training"
default_policy/is_training:0
 
C
observations3
default_policy/obs:0?????????TT
?
prev_action0
default_policy/prev_actions:0	?????????
?
prev_reward0
default_policy/prev_rewards:0?????????
+
timestep
default_policy/timestep:0	 E
action_dist_inputs/
default_policy/Squeeze:0??????????
action_logp0
default_policy/cond_2/Merge:0?????????6
action_prob'
default_policy/Exp:0?????????=
	actions_00
default_policy/cond_1/Merge:0	?????????7
vf_preds+
default_policy/Reshape:0?????????tensorflow/serving/predict