??4
??
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
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
?
MaxPoolWithArgmax

input"T
output"T
argmax"Targmax"
ksize	list(int)(0"
strides	list(int)(0"
Targmaxtype0	:
2	""
paddingstring:
SAMEVALID""
include_batch_in_indexbool( "
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
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
s
	ScatterNd
indices"Tindices
updates"T
shape"Tindices
output"T"	
Ttype"
Tindicestype:
2	
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
0
Sigmoid
x"T
y"T"
Ttype:

2
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
executor_typestring ?
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??/
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
: *
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
: *
dtype0
?
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namebatch_normalization/gamma
?
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
: *
dtype0
?
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namebatch_normalization/beta
?
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
: *
dtype0
?
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!batch_normalization/moving_mean
?
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
: *
dtype0
?
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#batch_normalization/moving_variance
?
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_1/gamma
?
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_1/beta
?
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:@*
dtype0
?
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_1/moving_mean
?
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:@*
dtype0
?
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_1/moving_variance
?
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_2/gamma
?
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_2/beta
?
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:@*
dtype0
?
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_2/moving_mean
?
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
?
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_2/moving_variance
?
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?* 
shared_nameconv2d_3/kernel
|
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*'
_output_shapes
:@?*
dtype0
s
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_3/bias
l
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_3/gamma
?
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namebatch_normalization_3/beta
?
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes	
:?*
dtype0
?
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!batch_normalization_3/moving_mean
?
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes	
:?*
dtype0
?
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%batch_normalization_3/moving_variance
?
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@* 
shared_nameconv2d_4/kernel
|
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*'
_output_shapes
:?@*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_4/gamma
?
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_4/beta
?
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
:@*
dtype0
?
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_4/moving_mean
?
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:@*
dtype0
?
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_4/moving_variance
?
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_5/gamma
?
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_5/beta
?
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
:@*
dtype0
?
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_5/moving_mean
?
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
:@*
dtype0
?
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_5/moving_variance
?
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ * 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
:@ *
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
: *
dtype0
?
batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_6/gamma
?
/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_6/beta
?
.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes
: *
dtype0
?
!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_6/moving_mean
?
5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes
: *
dtype0
?
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_6/moving_variance
?
9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
: *
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
:*
dtype0
?
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_7/gamma
?
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_7/beta
?
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
:*
dtype0
?
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_7/moving_mean
?
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
:*
dtype0
?
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_7/moving_variance
?
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes
:*
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
?
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d/kernel/m
?
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
: *
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
: *
dtype0
?
 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/batch_normalization/gamma/m
?
4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes
: *
dtype0
?
Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/batch_normalization/beta/m
?
3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_1/kernel/m
?
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:@*
dtype0
?
"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_1/gamma/m
?
6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes
:@*
dtype0
?
!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_1/beta/m
?
5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_2/kernel/m
?
*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*&
_output_shapes
:@@*
dtype0
?
Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_2/bias/m
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes
:@*
dtype0
?
"Adam/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_2/gamma/m
?
6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/m*
_output_shapes
:@*
dtype0
?
!Adam/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_2/beta/m
?
5Adam/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*'
shared_nameAdam/conv2d_3/kernel/m
?
*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*'
_output_shapes
:@?*
dtype0
?
Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/conv2d_3/bias/m
z
(Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/m*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_3/gamma/m
?
6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/m*
_output_shapes	
:?*
dtype0
?
!Adam/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/batch_normalization_3/beta/m
?
5Adam/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*'
shared_nameAdam/conv2d_4/kernel/m
?
*Adam/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/m*'
_output_shapes
:?@*
dtype0
?
Adam/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_4/bias/m
y
(Adam/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/m*
_output_shapes
:@*
dtype0
?
"Adam/batch_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_4/gamma/m
?
6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/m*
_output_shapes
:@*
dtype0
?
!Adam/batch_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_4/beta/m
?
5Adam/batch_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_5/kernel/m
?
*Adam/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/m*&
_output_shapes
:@@*
dtype0
?
Adam/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_5/bias/m
y
(Adam/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/m*
_output_shapes
:@*
dtype0
?
"Adam/batch_normalization_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_5/gamma/m
?
6Adam/batch_normalization_5/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/m*
_output_shapes
:@*
dtype0
?
!Adam/batch_normalization_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_5/beta/m
?
5Adam/batch_normalization_5/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *'
shared_nameAdam/conv2d_6/kernel/m
?
*Adam/conv2d_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/m*&
_output_shapes
:@ *
dtype0
?
Adam/conv2d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_6/bias/m
y
(Adam/conv2d_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/m*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_6/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_6/gamma/m
?
6Adam/batch_normalization_6/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_6/gamma/m*
_output_shapes
: *
dtype0
?
!Adam/batch_normalization_6/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_6/beta/m
?
5Adam/batch_normalization_6/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_6/beta/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_7/kernel/m
?
*Adam/conv2d_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_7/bias/m
y
(Adam/conv2d_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_7/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_7/gamma/m
?
6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/m*
_output_shapes
:*
dtype0
?
!Adam/batch_normalization_7/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_7/beta/m
?
5Adam/batch_normalization_7/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/m*
_output_shapes
:*
dtype0
?
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d/kernel/v
?
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
: *
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
: *
dtype0
?
 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/batch_normalization/gamma/v
?
4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes
: *
dtype0
?
Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/batch_normalization/beta/v
?
3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_1/kernel/v
?
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:@*
dtype0
?
"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_1/gamma/v
?
6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes
:@*
dtype0
?
!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_1/beta/v
?
5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_2/kernel/v
?
*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*&
_output_shapes
:@@*
dtype0
?
Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_2/bias/v
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes
:@*
dtype0
?
"Adam/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_2/gamma/v
?
6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/v*
_output_shapes
:@*
dtype0
?
!Adam/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_2/beta/v
?
5Adam/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*'
shared_nameAdam/conv2d_3/kernel/v
?
*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*'
_output_shapes
:@?*
dtype0
?
Adam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/conv2d_3/bias/v
z
(Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/v*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_3/gamma/v
?
6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/v*
_output_shapes	
:?*
dtype0
?
!Adam/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/batch_normalization_3/beta/v
?
5Adam/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*'
shared_nameAdam/conv2d_4/kernel/v
?
*Adam/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/v*'
_output_shapes
:?@*
dtype0
?
Adam/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_4/bias/v
y
(Adam/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/v*
_output_shapes
:@*
dtype0
?
"Adam/batch_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_4/gamma/v
?
6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/v*
_output_shapes
:@*
dtype0
?
!Adam/batch_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_4/beta/v
?
5Adam/batch_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_5/kernel/v
?
*Adam/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/v*&
_output_shapes
:@@*
dtype0
?
Adam/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_5/bias/v
y
(Adam/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/v*
_output_shapes
:@*
dtype0
?
"Adam/batch_normalization_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_5/gamma/v
?
6Adam/batch_normalization_5/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/v*
_output_shapes
:@*
dtype0
?
!Adam/batch_normalization_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_5/beta/v
?
5Adam/batch_normalization_5/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *'
shared_nameAdam/conv2d_6/kernel/v
?
*Adam/conv2d_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/v*&
_output_shapes
:@ *
dtype0
?
Adam/conv2d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_6/bias/v
y
(Adam/conv2d_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/v*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_6/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_6/gamma/v
?
6Adam/batch_normalization_6/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_6/gamma/v*
_output_shapes
: *
dtype0
?
!Adam/batch_normalization_6/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_6/beta/v
?
5Adam/batch_normalization_6/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_6/beta/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_7/kernel/v
?
*Adam/conv2d_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_7/bias/v
y
(Adam/conv2d_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_7/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_7/gamma/v
?
6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/v*
_output_shapes
:*
dtype0
?
!Adam/batch_normalization_7/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_7/beta/v
?
5Adam/batch_normalization_7/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer-15
layer-16
layer-17
layer_with_weights-8
layer-18
layer_with_weights-9
layer-19
layer-20
layer-21
layer_with_weights-10
layer-22
layer_with_weights-11
layer-23
layer-24
layer-25
layer_with_weights-12
layer-26
layer_with_weights-13
layer-27
layer-28
layer-29
layer_with_weights-14
layer-30
 layer_with_weights-15
 layer-31
!layer-32
"	optimizer
#regularization_losses
$trainable_variables
%	variables
&	keras_api
'
signatures
 
h

(kernel
)bias
*regularization_losses
+trainable_variables
,	variables
-	keras_api
?
.axis
	/gamma
0beta
1moving_mean
2moving_variance
3regularization_losses
4trainable_variables
5	variables
6	keras_api
R
7regularization_losses
8trainable_variables
9	variables
:	keras_api
R
;regularization_losses
<trainable_variables
=	variables
>	keras_api
h

?kernel
@bias
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
?
Eaxis
	Fgamma
Gbeta
Hmoving_mean
Imoving_variance
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api
R
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
R
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
h

Vkernel
Wbias
Xregularization_losses
Ytrainable_variables
Z	variables
[	keras_api
?
\axis
	]gamma
^beta
_moving_mean
`moving_variance
aregularization_losses
btrainable_variables
c	variables
d	keras_api
R
eregularization_losses
ftrainable_variables
g	variables
h	keras_api
R
iregularization_losses
jtrainable_variables
k	variables
l	keras_api
h

mkernel
nbias
oregularization_losses
ptrainable_variables
q	variables
r	keras_api
?
saxis
	tgamma
ubeta
vmoving_mean
wmoving_variance
xregularization_losses
ytrainable_variables
z	variables
{	keras_api
R
|regularization_losses
}trainable_variables
~	variables
	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate(m?)m?/m?0m??m?@m?Fm?Gm?Vm?Wm?]m?^m?mm?nm?tm?um?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?(v?)v?/v?0v??v?@v?Fv?Gv?Vv?Wv?]v?^v?mv?nv?tv?uv?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?
 
?
(0
)1
/2
03
?4
@5
F6
G7
V8
W9
]10
^11
m12
n13
t14
u15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?
(0
)1
/2
03
14
25
?6
@7
F8
G9
H10
I11
V12
W13
]14
^15
_16
`17
m18
n19
t20
u21
v22
w23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?
#regularization_losses
?layer_metrics
$trainable_variables
 ?layer_regularization_losses
%	variables
?metrics
?non_trainable_variables
?layers
 
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

(0
)1

(0
)1
?
*regularization_losses
?layer_metrics
+trainable_variables
 ?layer_regularization_losses
,	variables
?metrics
?non_trainable_variables
?layers
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

/0
01

/0
01
12
23
?
3regularization_losses
?layer_metrics
4trainable_variables
 ?layer_regularization_losses
5	variables
?metrics
?non_trainable_variables
?layers
 
 
 
?
7regularization_losses
?layer_metrics
8trainable_variables
 ?layer_regularization_losses
9	variables
?metrics
?non_trainable_variables
?layers
 
 
 
?
;regularization_losses
?layer_metrics
<trainable_variables
 ?layer_regularization_losses
=	variables
?metrics
?non_trainable_variables
?layers
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
@1

?0
@1
?
Aregularization_losses
?layer_metrics
Btrainable_variables
 ?layer_regularization_losses
C	variables
?metrics
?non_trainable_variables
?layers
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

F0
G1

F0
G1
H2
I3
?
Jregularization_losses
?layer_metrics
Ktrainable_variables
 ?layer_regularization_losses
L	variables
?metrics
?non_trainable_variables
?layers
 
 
 
?
Nregularization_losses
?layer_metrics
Otrainable_variables
 ?layer_regularization_losses
P	variables
?metrics
?non_trainable_variables
?layers
 
 
 
?
Rregularization_losses
?layer_metrics
Strainable_variables
 ?layer_regularization_losses
T	variables
?metrics
?non_trainable_variables
?layers
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

V0
W1

V0
W1
?
Xregularization_losses
?layer_metrics
Ytrainable_variables
 ?layer_regularization_losses
Z	variables
?metrics
?non_trainable_variables
?layers
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

]0
^1

]0
^1
_2
`3
?
aregularization_losses
?layer_metrics
btrainable_variables
 ?layer_regularization_losses
c	variables
?metrics
?non_trainable_variables
?layers
 
 
 
?
eregularization_losses
?layer_metrics
ftrainable_variables
 ?layer_regularization_losses
g	variables
?metrics
?non_trainable_variables
?layers
 
 
 
?
iregularization_losses
?layer_metrics
jtrainable_variables
 ?layer_regularization_losses
k	variables
?metrics
?non_trainable_variables
?layers
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

m0
n1

m0
n1
?
oregularization_losses
?layer_metrics
ptrainable_variables
 ?layer_regularization_losses
q	variables
?metrics
?non_trainable_variables
?layers
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

t0
u1

t0
u1
v2
w3
?
xregularization_losses
?layer_metrics
ytrainable_variables
 ?layer_regularization_losses
z	variables
?metrics
?non_trainable_variables
?layers
 
 
 
?
|regularization_losses
?layer_metrics
}trainable_variables
 ?layer_regularization_losses
~	variables
?metrics
?non_trainable_variables
?layers
 
 
 
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
 
 
 
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
[Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
 
fd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
 
?0
?1
?2
?3
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
 
 
 
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
 
 
 
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
\Z
VARIABLE_VALUEconv2d_5/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_5/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
 
ge
VARIABLE_VALUEbatch_normalization_5/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_5/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_5/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_5/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
 
?0
?1
?2
?3
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
 
 
 
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
 
 
 
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
\Z
VARIABLE_VALUEconv2d_6/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_6/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
 
ge
VARIABLE_VALUEbatch_normalization_6/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_6/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_6/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_6/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
 
?0
?1
?2
?3
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
 
 
 
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
 
 
 
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
\Z
VARIABLE_VALUEconv2d_7/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_7/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
 
ge
VARIABLE_VALUEbatch_normalization_7/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_7/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_7/moving_mean<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_7/moving_variance@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
 
?0
?1
?2
?3
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
 
 
 
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0
?1
~
10
21
H2
I3
_4
`5
v6
w7
?8
?9
?10
?11
?12
?13
?14
?15
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
 
 
 
 
 
 
 
 

10
21
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

H0
I1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

_0
`1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

v0
w1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
|z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/batch_normalization/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/batch_normalization/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization_1/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization_2/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization_3/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_4/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_4/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_4/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization_4/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_5/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_5/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_5/gamma/mRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization_5/beta/mQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_6/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_6/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_6/gamma/mRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization_6/beta/mQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_7/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_7/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_7/gamma/mRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization_7/beta/mQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/batch_normalization/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/batch_normalization/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization_1/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization_2/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization_3/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_4/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_4/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_4/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization_4/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_5/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_5/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_5/gamma/vRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization_5/beta/vQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_6/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_6/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_6/gamma/vRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization_6/beta/vQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_7/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_7/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_7/gamma/vRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization_7/beta/vQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_3/kernelconv2d_3/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_4/kernelconv2d_4/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv2d_5/kernelconv2d_5/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_6/kernelconv2d_6/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_7/kernelconv2d_7/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_variance*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_8515
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?/
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_2/beta/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d_3/bias/m/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_3/beta/m/Read/ReadVariableOp*Adam/conv2d_4/kernel/m/Read/ReadVariableOp(Adam/conv2d_4/bias/m/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_4/beta/m/Read/ReadVariableOp*Adam/conv2d_5/kernel/m/Read/ReadVariableOp(Adam/conv2d_5/bias/m/Read/ReadVariableOp6Adam/batch_normalization_5/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_5/beta/m/Read/ReadVariableOp*Adam/conv2d_6/kernel/m/Read/ReadVariableOp(Adam/conv2d_6/bias/m/Read/ReadVariableOp6Adam/batch_normalization_6/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_6/beta/m/Read/ReadVariableOp*Adam/conv2d_7/kernel/m/Read/ReadVariableOp(Adam/conv2d_7/bias/m/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_7/beta/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_2/beta/v/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOp(Adam/conv2d_3/bias/v/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_3/beta/v/Read/ReadVariableOp*Adam/conv2d_4/kernel/v/Read/ReadVariableOp(Adam/conv2d_4/bias/v/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_4/beta/v/Read/ReadVariableOp*Adam/conv2d_5/kernel/v/Read/ReadVariableOp(Adam/conv2d_5/bias/v/Read/ReadVariableOp6Adam/batch_normalization_5/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_5/beta/v/Read/ReadVariableOp*Adam/conv2d_6/kernel/v/Read/ReadVariableOp(Adam/conv2d_6/bias/v/Read/ReadVariableOp6Adam/batch_normalization_6/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_6/beta/v/Read/ReadVariableOp*Adam/conv2d_7/kernel/v/Read/ReadVariableOp(Adam/conv2d_7/bias/v/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_7/beta/v/Read/ReadVariableOpConst*?
Tin
}2{	*
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
GPU 2J 8? *'
f"R 
__inference__traced_save_11327
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_3/kernelconv2d_3/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_4/kernelconv2d_4/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv2d_5/kernelconv2d_5/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_6/kernelconv2d_6/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_7/kernelconv2d_7/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_variance	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d/kernel/mAdam/conv2d/bias/m Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/m"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/m"Adam/batch_normalization_2/gamma/m!Adam/batch_normalization_2/beta/mAdam/conv2d_3/kernel/mAdam/conv2d_3/bias/m"Adam/batch_normalization_3/gamma/m!Adam/batch_normalization_3/beta/mAdam/conv2d_4/kernel/mAdam/conv2d_4/bias/m"Adam/batch_normalization_4/gamma/m!Adam/batch_normalization_4/beta/mAdam/conv2d_5/kernel/mAdam/conv2d_5/bias/m"Adam/batch_normalization_5/gamma/m!Adam/batch_normalization_5/beta/mAdam/conv2d_6/kernel/mAdam/conv2d_6/bias/m"Adam/batch_normalization_6/gamma/m!Adam/batch_normalization_6/beta/mAdam/conv2d_7/kernel/mAdam/conv2d_7/bias/m"Adam/batch_normalization_7/gamma/m!Adam/batch_normalization_7/beta/mAdam/conv2d/kernel/vAdam/conv2d/bias/v Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/v"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/v"Adam/batch_normalization_2/gamma/v!Adam/batch_normalization_2/beta/vAdam/conv2d_3/kernel/vAdam/conv2d_3/bias/v"Adam/batch_normalization_3/gamma/v!Adam/batch_normalization_3/beta/vAdam/conv2d_4/kernel/vAdam/conv2d_4/bias/v"Adam/batch_normalization_4/gamma/v!Adam/batch_normalization_4/beta/vAdam/conv2d_5/kernel/vAdam/conv2d_5/bias/v"Adam/batch_normalization_5/gamma/v!Adam/batch_normalization_5/beta/vAdam/conv2d_6/kernel/vAdam/conv2d_6/bias/v"Adam/batch_normalization_6/gamma/v!Adam/batch_normalization_6/beta/vAdam/conv2d_7/kernel/vAdam/conv2d_7/bias/v"Adam/batch_normalization_7/gamma/v!Adam/batch_normalization_7/beta/v*?
Tin~
|2z*
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
GPU 2J 8? **
f%R#
!__inference__traced_restore_11700??*
?
?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9617

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
U__inference_max_pooling_with_argmax2d_2_layer_call_and_return_conditional_losses_6547

inputs
identity

identity_1?
MaxPoolWithArgmaxMaxPoolWithArgmaxinputs*
T0*J
_output_shapes8
6:?????????  @:?????????  @*
ksize
*
paddingSAME*
strides
2
MaxPoolWithArgmaxy
CastCastMaxPoolWithArgmax:argmax:0*

DstT0*

SrcT0	*/
_output_shapes
:?????????  @2
Castv
IdentityIdentityMaxPoolWithArgmax:output:0*
T0*/
_output_shapes
:?????????  @2

Identityh

Identity_1IdentityCast:y:0*
T0*/
_output_shapes
:?????????  @2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@@:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6461

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_7268

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?+
u
K__inference_max_unpooling2d_2_layer_call_and_return_conditional_losses_6847

inputs
inputs_1
identityg
CastCastinputs_1*

DstT0*

SrcT0*/
_output_shapes
:?????????@@@2
CastD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2T
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapem
ReshapeReshapeCast:y:0Reshape/shape:output:0*
T0*#
_output_shapes
:?????????2	
Reshapek
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim?

ExpandDims
ExpandDimsReshape:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2

ExpandDimsu
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_1/shapeq
	Reshape_1ReshapeinputsReshape_1/shape:output:0*
T0*#
_output_shapes
:?????????2
	Reshape_1?
Rank/packedPackstrided_slice:output:0mul:z:0	mul_1:z:0strided_slice_3:output:0*
N*
T0*
_output_shapes
:2
Rank/packedN
RankConst*
_output_shapes
: *
dtype0*
value	B :2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltan
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:2
range?

Prod/inputPackstrided_slice:output:0mul:z:0	mul_1:z:0strided_slice_3:output:0*
N*
T0*
_output_shapes
:2

Prod/inputZ
ProdProdProd/input:output:0range:output:0*
T0*
_output_shapes
: 2
Prodg
ScatterNd/shapePackProd:output:0*
N*
T0*
_output_shapes
:2
ScatterNd/shape?
	ScatterNd	ScatterNdExpandDims:output:0Reshape_1:output:0ScatterNd/shape:output:0*
T0*
Tindices0*#
_output_shapes
:?????????2
	ScatterNd{
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2
Reshape_2/shape?
	Reshape_2ReshapeScatterNd:output:0Reshape_2/shape:output:0*
T0*1
_output_shapes
:???????????@2
	Reshape_2p
IdentityIdentityReshape_2:output:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????@@@:?????????@@@:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_1_layer_call_fn_9710

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_64612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
H
,__inference_activation_3_layer_call_fn_10099

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_65982
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????  ?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_4_layer_call_fn_10283

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_58782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
b
F__inference_activation_5_layer_call_and_return_conditional_losses_6801

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@@:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_6_layer_call_fn_10689

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_61302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
2__inference_batch_normalization_layer_call_fn_9540

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_76742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_10037

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????  ?: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_10270

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  @2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10676

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
5__inference_batch_normalization_4_layer_call_fn_10309

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_66902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
C__inference_conv2d_5_layer_call_and_return_conditional_losses_10392

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
(__inference_conv2d_6_layer_call_fn_10604

inputs!
unknown:@ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_68592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
B__inference_conv2d_3_layer_call_and_return_conditional_losses_6560

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?2	
BiasAddt
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_2_layer_call_fn_9906

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_75082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5922

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_9488

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_6130

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10843

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
(__inference_conv2d_7_layer_call_fn_10807

inputs!
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_7_layer_call_and_return_conditional_losses_69552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6978

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
(__inference_conv2d_5_layer_call_fn_10401

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_67632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
??
?
?__inference_model_layer_call_and_return_conditional_losses_8406
input_1%
conv2d_8273: 
conv2d_8275: &
batch_normalization_8278: &
batch_normalization_8280: &
batch_normalization_8282: &
batch_normalization_8284: '
conv2d_1_8290: @
conv2d_1_8292:@(
batch_normalization_1_8295:@(
batch_normalization_1_8297:@(
batch_normalization_1_8299:@(
batch_normalization_1_8301:@'
conv2d_2_8307:@@
conv2d_2_8309:@(
batch_normalization_2_8312:@(
batch_normalization_2_8314:@(
batch_normalization_2_8316:@(
batch_normalization_2_8318:@(
conv2d_3_8324:@?
conv2d_3_8326:	?)
batch_normalization_3_8329:	?)
batch_normalization_3_8331:	?)
batch_normalization_3_8333:	?)
batch_normalization_3_8335:	?(
conv2d_4_8342:?@
conv2d_4_8344:@(
batch_normalization_4_8347:@(
batch_normalization_4_8349:@(
batch_normalization_4_8351:@(
batch_normalization_4_8353:@'
conv2d_5_8358:@@
conv2d_5_8360:@(
batch_normalization_5_8363:@(
batch_normalization_5_8365:@(
batch_normalization_5_8367:@(
batch_normalization_5_8369:@'
conv2d_6_8374:@ 
conv2d_6_8376: (
batch_normalization_6_8379: (
batch_normalization_6_8381: (
batch_normalization_6_8383: (
batch_normalization_6_8385: '
conv2d_7_8390: 
conv2d_7_8392:(
batch_normalization_7_8395:(
batch_normalization_7_8397:(
batch_normalization_7_8399:(
batch_normalization_7_8401:
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?-batch_normalization_5/StatefulPartitionedCall?-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_8273conv2d_8275*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_63772 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_8278batch_normalization_8280batch_normalization_8282batch_normalization_8284*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_76742-
+batch_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_64152
activation/PartitionedCall?
)max_pooling_with_argmax2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::??????????? :??????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_max_pooling_with_argmax2d_layer_call_and_return_conditional_losses_76302+
)max_pooling_with_argmax2d/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall2max_pooling_with_argmax2d/PartitionedCall:output:0conv2d_1_8290conv2d_1_8292*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_64382"
 conv2d_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_8295batch_normalization_1_8297batch_normalization_1_8299batch_normalization_1_8301*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_75912/
-batch_normalization_1/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_64762
activation_1/PartitionedCall?
+max_pooling_with_argmax2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:?????????@@@:?????????@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_max_pooling_with_argmax2d_1_layer_call_and_return_conditional_losses_75472-
+max_pooling_with_argmax2d_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall4max_pooling_with_argmax2d_1/PartitionedCall:output:0conv2d_2_8307conv2d_2_8309*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_64992"
 conv2d_2/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_8312batch_normalization_2_8314batch_normalization_2_8316batch_normalization_2_8318*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_75082/
-batch_normalization_2/StatefulPartitionedCall?
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_65372
activation_2/PartitionedCall?
+max_pooling_with_argmax2d_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:?????????  @:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_max_pooling_with_argmax2d_2_layer_call_and_return_conditional_losses_74642-
+max_pooling_with_argmax2d_2/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall4max_pooling_with_argmax2d_2/PartitionedCall:output:0conv2d_3_8324conv2d_3_8326*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_65602"
 conv2d_3/StatefulPartitionedCall?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_8329batch_normalization_3_8331batch_normalization_3_8333batch_normalization_3_8335*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_74252/
-batch_normalization_3/StatefulPartitionedCall?
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_65982
activation_3/PartitionedCall?
+max_pooling_with_argmax2d_3/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *L
_output_shapes:
8:??????????:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_max_pooling_with_argmax2d_3_layer_call_and_return_conditional_losses_73812-
+max_pooling_with_argmax2d_3/PartitionedCall?
max_unpooling2d/PartitionedCallPartitionedCall4max_pooling_with_argmax2d_3/PartitionedCall:output:04max_pooling_with_argmax2d_3/PartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_unpooling2d_layer_call_and_return_conditional_losses_66552!
max_unpooling2d/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_unpooling2d/PartitionedCall:output:0conv2d_4_8342conv2d_4_8344*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_66672"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_8347batch_normalization_4_8349batch_normalization_4_8351batch_normalization_4_8353*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_73352/
-batch_normalization_4/StatefulPartitionedCall?
activation_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_67052
activation_4/PartitionedCall?
!max_unpooling2d_1/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:04max_pooling_with_argmax2d_2/PartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_unpooling2d_1_layer_call_and_return_conditional_losses_67512#
!max_unpooling2d_1/PartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall*max_unpooling2d_1/PartitionedCall:output:0conv2d_5_8358conv2d_5_8360*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_67632"
 conv2d_5/StatefulPartitionedCall?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_8363batch_normalization_5_8365batch_normalization_5_8367batch_normalization_5_8369*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_72682/
-batch_normalization_5/StatefulPartitionedCall?
activation_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_5_layer_call_and_return_conditional_losses_68012
activation_5/PartitionedCall?
!max_unpooling2d_2/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:04max_pooling_with_argmax2d_1/PartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_unpooling2d_2_layer_call_and_return_conditional_losses_68472#
!max_unpooling2d_2/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall*max_unpooling2d_2/PartitionedCall:output:0conv2d_6_8374conv2d_6_8376*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_68592"
 conv2d_6/StatefulPartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_8379batch_normalization_6_8381batch_normalization_6_8383batch_normalization_6_8385*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_72012/
-batch_normalization_6/StatefulPartitionedCall?
activation_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_6_layer_call_and_return_conditional_losses_68972
activation_6/PartitionedCall?
!max_unpooling2d_3/PartitionedCallPartitionedCall%activation_6/PartitionedCall:output:02max_pooling_with_argmax2d/PartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_unpooling2d_3_layer_call_and_return_conditional_losses_69432#
!max_unpooling2d_3/PartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall*max_unpooling2d_3/PartitionedCall:output:0conv2d_7_8390conv2d_7_8392*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_7_layer_call_and_return_conditional_losses_69552"
 conv2d_7/StatefulPartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_7_8395batch_normalization_7_8397batch_normalization_7_8399batch_normalization_7_8401*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_71342/
-batch_normalization_7/StatefulPartitionedCall?
activation_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_7_layer_call_and_return_conditional_losses_69932
activation_7/PartitionedCall?
IdentityIdentity%activation_7/PartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes
}:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_7201

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
??
?*
?__inference_model_layer_call_and_return_conditional_losses_8855

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 9
+batch_normalization_readvariableop_resource: ;
-batch_normalization_readvariableop_1_resource: J
<batch_normalization_fusedbatchnormv3_readvariableop_resource: L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_1_conv2d_readvariableop_resource: @6
(conv2d_1_biasadd_readvariableop_resource:@;
-batch_normalization_1_readvariableop_resource:@=
/batch_normalization_1_readvariableop_1_resource:@L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@A
'conv2d_2_conv2d_readvariableop_resource:@@6
(conv2d_2_biasadd_readvariableop_resource:@;
-batch_normalization_2_readvariableop_resource:@=
/batch_normalization_2_readvariableop_1_resource:@L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@B
'conv2d_3_conv2d_readvariableop_resource:@?7
(conv2d_3_biasadd_readvariableop_resource:	?<
-batch_normalization_3_readvariableop_resource:	?>
/batch_normalization_3_readvariableop_1_resource:	?M
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	?O
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	?B
'conv2d_4_conv2d_readvariableop_resource:?@6
(conv2d_4_biasadd_readvariableop_resource:@;
-batch_normalization_4_readvariableop_resource:@=
/batch_normalization_4_readvariableop_1_resource:@L
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:@A
'conv2d_5_conv2d_readvariableop_resource:@@6
(conv2d_5_biasadd_readvariableop_resource:@;
-batch_normalization_5_readvariableop_resource:@=
/batch_normalization_5_readvariableop_1_resource:@L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@A
'conv2d_6_conv2d_readvariableop_resource:@ 6
(conv2d_6_biasadd_readvariableop_resource: ;
-batch_normalization_6_readvariableop_resource: =
/batch_normalization_6_readvariableop_1_resource: L
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_7_conv2d_readvariableop_resource: 6
(conv2d_7_biasadd_readvariableop_resource:;
-batch_normalization_7_readvariableop_resource:=
/batch_normalization_7_readvariableop_1_resource:L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:
identity??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_2/ReadVariableOp?&batch_normalization_2/ReadVariableOp_1?5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_3/ReadVariableOp?&batch_normalization_3/ReadVariableOp_1?5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_4/ReadVariableOp?&batch_normalization_4/ReadVariableOp_1?5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_5/ReadVariableOp?&batch_normalization_5/ReadVariableOp_1?5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_6/ReadVariableOp?&batch_normalization_6/ReadVariableOp_1?5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_7/ReadVariableOp?&batch_normalization_7/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2&
$batch_normalization/FusedBatchNormV3?
activation/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
activation/Relu?
+max_pooling_with_argmax2d/MaxPoolWithArgmaxMaxPoolWithArgmaxactivation/Relu:activations:0*
T0*N
_output_shapes<
::??????????? :??????????? *
ksize
*
paddingSAME*
strides
2-
+max_pooling_with_argmax2d/MaxPoolWithArgmax?
max_pooling_with_argmax2d/CastCast4max_pooling_with_argmax2d/MaxPoolWithArgmax:argmax:0*

DstT0*

SrcT0	*1
_output_shapes
:??????????? 2 
max_pooling_with_argmax2d/Cast?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2D4max_pooling_with_argmax2d/MaxPoolWithArgmax:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
conv2d_1/BiasAdd?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3?
activation_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????@2
activation_1/Relu?
-max_pooling_with_argmax2d_1/MaxPoolWithArgmaxMaxPoolWithArgmaxactivation_1/Relu:activations:0*
T0*J
_output_shapes8
6:?????????@@@:?????????@@@*
ksize
*
paddingSAME*
strides
2/
-max_pooling_with_argmax2d_1/MaxPoolWithArgmax?
 max_pooling_with_argmax2d_1/CastCast6max_pooling_with_argmax2d_1/MaxPoolWithArgmax:argmax:0*

DstT0*

SrcT0	*/
_output_shapes
:?????????@@@2"
 max_pooling_with_argmax2d_1/Cast?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D6max_pooling_with_argmax2d_1/MaxPoolWithArgmax:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@2
conv2d_2/BiasAdd?
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp?
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1?
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3?
activation_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@@2
activation_2/Relu?
-max_pooling_with_argmax2d_2/MaxPoolWithArgmaxMaxPoolWithArgmaxactivation_2/Relu:activations:0*
T0*J
_output_shapes8
6:?????????  @:?????????  @*
ksize
*
paddingSAME*
strides
2/
-max_pooling_with_argmax2d_2/MaxPoolWithArgmax?
 max_pooling_with_argmax2d_2/CastCast6max_pooling_with_argmax2d_2/MaxPoolWithArgmax:argmax:0*

DstT0*

SrcT0	*/
_output_shapes
:?????????  @2"
 max_pooling_with_argmax2d_2/Cast?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2D6max_pooling_with_argmax2d_2/MaxPoolWithArgmax:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?2
conv2d_3/BiasAdd?
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_3/ReadVariableOp?
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_3/ReadVariableOp_1?
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3?
activation_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????  ?2
activation_3/Relu?
-max_pooling_with_argmax2d_3/MaxPoolWithArgmaxMaxPoolWithArgmaxactivation_3/Relu:activations:0*
T0*L
_output_shapes:
8:??????????:??????????*
ksize
*
paddingSAME*
strides
2/
-max_pooling_with_argmax2d_3/MaxPoolWithArgmax?
 max_pooling_with_argmax2d_3/CastCast6max_pooling_with_argmax2d_3/MaxPoolWithArgmax:argmax:0*

DstT0*

SrcT0	*0
_output_shapes
:??????????2"
 max_pooling_with_argmax2d_3/Cast?
max_unpooling2d/CastCast$max_pooling_with_argmax2d_3/Cast:y:0*

DstT0*

SrcT0*0
_output_shapes
:??????????2
max_unpooling2d/Cast?
max_unpooling2d/ShapeShape6max_pooling_with_argmax2d_3/MaxPoolWithArgmax:output:0*
T0*
_output_shapes
:2
max_unpooling2d/Shape?
#max_unpooling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#max_unpooling2d/strided_slice/stack?
%max_unpooling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%max_unpooling2d/strided_slice/stack_1?
%max_unpooling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%max_unpooling2d/strided_slice/stack_2?
max_unpooling2d/strided_sliceStridedSlicemax_unpooling2d/Shape:output:0,max_unpooling2d/strided_slice/stack:output:0.max_unpooling2d/strided_slice/stack_1:output:0.max_unpooling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
max_unpooling2d/strided_slice?
%max_unpooling2d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2'
%max_unpooling2d/strided_slice_1/stack?
'max_unpooling2d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d/strided_slice_1/stack_1?
'max_unpooling2d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d/strided_slice_1/stack_2?
max_unpooling2d/strided_slice_1StridedSlicemax_unpooling2d/Shape:output:0.max_unpooling2d/strided_slice_1/stack:output:00max_unpooling2d/strided_slice_1/stack_1:output:00max_unpooling2d/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
max_unpooling2d/strided_slice_1p
max_unpooling2d/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d/mul/y?
max_unpooling2d/mulMul(max_unpooling2d/strided_slice_1:output:0max_unpooling2d/mul/y:output:0*
T0*
_output_shapes
: 2
max_unpooling2d/mul?
%max_unpooling2d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2'
%max_unpooling2d/strided_slice_2/stack?
'max_unpooling2d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d/strided_slice_2/stack_1?
'max_unpooling2d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d/strided_slice_2/stack_2?
max_unpooling2d/strided_slice_2StridedSlicemax_unpooling2d/Shape:output:0.max_unpooling2d/strided_slice_2/stack:output:00max_unpooling2d/strided_slice_2/stack_1:output:00max_unpooling2d/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
max_unpooling2d/strided_slice_2t
max_unpooling2d/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d/mul_1/y?
max_unpooling2d/mul_1Mul(max_unpooling2d/strided_slice_2:output:0 max_unpooling2d/mul_1/y:output:0*
T0*
_output_shapes
: 2
max_unpooling2d/mul_1?
%max_unpooling2d/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2'
%max_unpooling2d/strided_slice_3/stack?
'max_unpooling2d/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d/strided_slice_3/stack_1?
'max_unpooling2d/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d/strided_slice_3/stack_2?
max_unpooling2d/strided_slice_3StridedSlicemax_unpooling2d/Shape:output:0.max_unpooling2d/strided_slice_3/stack:output:00max_unpooling2d/strided_slice_3/stack_1:output:00max_unpooling2d/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
max_unpooling2d/strided_slice_3?
max_unpooling2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
max_unpooling2d/Reshape/shape?
max_unpooling2d/ReshapeReshapemax_unpooling2d/Cast:y:0&max_unpooling2d/Reshape/shape:output:0*
T0*#
_output_shapes
:?????????2
max_unpooling2d/Reshape?
max_unpooling2d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
max_unpooling2d/ExpandDims/dim?
max_unpooling2d/ExpandDims
ExpandDims max_unpooling2d/Reshape:output:0'max_unpooling2d/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2
max_unpooling2d/ExpandDims?
max_unpooling2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2!
max_unpooling2d/Reshape_1/shape?
max_unpooling2d/Reshape_1Reshape6max_pooling_with_argmax2d_3/MaxPoolWithArgmax:output:0(max_unpooling2d/Reshape_1/shape:output:0*
T0*#
_output_shapes
:?????????2
max_unpooling2d/Reshape_1?
max_unpooling2d/Rank/packedPack&max_unpooling2d/strided_slice:output:0max_unpooling2d/mul:z:0max_unpooling2d/mul_1:z:0(max_unpooling2d/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2
max_unpooling2d/Rank/packedn
max_unpooling2d/RankConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d/Rank|
max_unpooling2d/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
max_unpooling2d/range/start|
max_unpooling2d/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d/range/delta?
max_unpooling2d/rangeRange$max_unpooling2d/range/start:output:0max_unpooling2d/Rank:output:0$max_unpooling2d/range/delta:output:0*
_output_shapes
:2
max_unpooling2d/range?
max_unpooling2d/Prod/inputPack&max_unpooling2d/strided_slice:output:0max_unpooling2d/mul:z:0max_unpooling2d/mul_1:z:0(max_unpooling2d/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2
max_unpooling2d/Prod/input?
max_unpooling2d/ProdProd#max_unpooling2d/Prod/input:output:0max_unpooling2d/range:output:0*
T0*
_output_shapes
: 2
max_unpooling2d/Prod?
max_unpooling2d/ScatterNd/shapePackmax_unpooling2d/Prod:output:0*
N*
T0*
_output_shapes
:2!
max_unpooling2d/ScatterNd/shape?
max_unpooling2d/ScatterNd	ScatterNd#max_unpooling2d/ExpandDims:output:0"max_unpooling2d/Reshape_1:output:0(max_unpooling2d/ScatterNd/shape:output:0*
T0*
Tindices0*#
_output_shapes
:?????????2
max_unpooling2d/ScatterNd?
max_unpooling2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????        ?   2!
max_unpooling2d/Reshape_2/shape?
max_unpooling2d/Reshape_2Reshape"max_unpooling2d/ScatterNd:output:0(max_unpooling2d/Reshape_2/shape:output:0*
T0*0
_output_shapes
:?????????  ?2
max_unpooling2d/Reshape_2?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2D"max_unpooling2d/Reshape_2:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2
conv2d_4/BiasAdd?
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_4/ReadVariableOp?
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_4/ReadVariableOp_1?
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3?
activation_4/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  @2
activation_4/Relu?
max_unpooling2d_1/CastCast$max_pooling_with_argmax2d_2/Cast:y:0*

DstT0*

SrcT0*/
_output_shapes
:?????????  @2
max_unpooling2d_1/Cast?
max_unpooling2d_1/ShapeShapeactivation_4/Relu:activations:0*
T0*
_output_shapes
:2
max_unpooling2d_1/Shape?
%max_unpooling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%max_unpooling2d_1/strided_slice/stack?
'max_unpooling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d_1/strided_slice/stack_1?
'max_unpooling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d_1/strided_slice/stack_2?
max_unpooling2d_1/strided_sliceStridedSlice max_unpooling2d_1/Shape:output:0.max_unpooling2d_1/strided_slice/stack:output:00max_unpooling2d_1/strided_slice/stack_1:output:00max_unpooling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
max_unpooling2d_1/strided_slice?
'max_unpooling2d_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d_1/strided_slice_1/stack?
)max_unpooling2d_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_1/strided_slice_1/stack_1?
)max_unpooling2d_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_1/strided_slice_1/stack_2?
!max_unpooling2d_1/strided_slice_1StridedSlice max_unpooling2d_1/Shape:output:00max_unpooling2d_1/strided_slice_1/stack:output:02max_unpooling2d_1/strided_slice_1/stack_1:output:02max_unpooling2d_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!max_unpooling2d_1/strided_slice_1t
max_unpooling2d_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d_1/mul/y?
max_unpooling2d_1/mulMul*max_unpooling2d_1/strided_slice_1:output:0 max_unpooling2d_1/mul/y:output:0*
T0*
_output_shapes
: 2
max_unpooling2d_1/mul?
'max_unpooling2d_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d_1/strided_slice_2/stack?
)max_unpooling2d_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_1/strided_slice_2/stack_1?
)max_unpooling2d_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_1/strided_slice_2/stack_2?
!max_unpooling2d_1/strided_slice_2StridedSlice max_unpooling2d_1/Shape:output:00max_unpooling2d_1/strided_slice_2/stack:output:02max_unpooling2d_1/strided_slice_2/stack_1:output:02max_unpooling2d_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!max_unpooling2d_1/strided_slice_2x
max_unpooling2d_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d_1/mul_1/y?
max_unpooling2d_1/mul_1Mul*max_unpooling2d_1/strided_slice_2:output:0"max_unpooling2d_1/mul_1/y:output:0*
T0*
_output_shapes
: 2
max_unpooling2d_1/mul_1?
'max_unpooling2d_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d_1/strided_slice_3/stack?
)max_unpooling2d_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_1/strided_slice_3/stack_1?
)max_unpooling2d_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_1/strided_slice_3/stack_2?
!max_unpooling2d_1/strided_slice_3StridedSlice max_unpooling2d_1/Shape:output:00max_unpooling2d_1/strided_slice_3/stack:output:02max_unpooling2d_1/strided_slice_3/stack_1:output:02max_unpooling2d_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!max_unpooling2d_1/strided_slice_3?
max_unpooling2d_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2!
max_unpooling2d_1/Reshape/shape?
max_unpooling2d_1/ReshapeReshapemax_unpooling2d_1/Cast:y:0(max_unpooling2d_1/Reshape/shape:output:0*
T0*#
_output_shapes
:?????????2
max_unpooling2d_1/Reshape?
 max_unpooling2d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 max_unpooling2d_1/ExpandDims/dim?
max_unpooling2d_1/ExpandDims
ExpandDims"max_unpooling2d_1/Reshape:output:0)max_unpooling2d_1/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2
max_unpooling2d_1/ExpandDims?
!max_unpooling2d_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2#
!max_unpooling2d_1/Reshape_1/shape?
max_unpooling2d_1/Reshape_1Reshapeactivation_4/Relu:activations:0*max_unpooling2d_1/Reshape_1/shape:output:0*
T0*#
_output_shapes
:?????????2
max_unpooling2d_1/Reshape_1?
max_unpooling2d_1/Rank/packedPack(max_unpooling2d_1/strided_slice:output:0max_unpooling2d_1/mul:z:0max_unpooling2d_1/mul_1:z:0*max_unpooling2d_1/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2
max_unpooling2d_1/Rank/packedr
max_unpooling2d_1/RankConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d_1/Rank?
max_unpooling2d_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
max_unpooling2d_1/range/start?
max_unpooling2d_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d_1/range/delta?
max_unpooling2d_1/rangeRange&max_unpooling2d_1/range/start:output:0max_unpooling2d_1/Rank:output:0&max_unpooling2d_1/range/delta:output:0*
_output_shapes
:2
max_unpooling2d_1/range?
max_unpooling2d_1/Prod/inputPack(max_unpooling2d_1/strided_slice:output:0max_unpooling2d_1/mul:z:0max_unpooling2d_1/mul_1:z:0*max_unpooling2d_1/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2
max_unpooling2d_1/Prod/input?
max_unpooling2d_1/ProdProd%max_unpooling2d_1/Prod/input:output:0 max_unpooling2d_1/range:output:0*
T0*
_output_shapes
: 2
max_unpooling2d_1/Prod?
!max_unpooling2d_1/ScatterNd/shapePackmax_unpooling2d_1/Prod:output:0*
N*
T0*
_output_shapes
:2#
!max_unpooling2d_1/ScatterNd/shape?
max_unpooling2d_1/ScatterNd	ScatterNd%max_unpooling2d_1/ExpandDims:output:0$max_unpooling2d_1/Reshape_1:output:0*max_unpooling2d_1/ScatterNd/shape:output:0*
T0*
Tindices0*#
_output_shapes
:?????????2
max_unpooling2d_1/ScatterNd?
!max_unpooling2d_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????@   @   @   2#
!max_unpooling2d_1/Reshape_2/shape?
max_unpooling2d_1/Reshape_2Reshape$max_unpooling2d_1/ScatterNd:output:0*max_unpooling2d_1/Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????@@@2
max_unpooling2d_1/Reshape_2?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2D$max_unpooling2d_1/Reshape_2:output:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@2
conv2d_5/BiasAdd?
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_5/ReadVariableOp?
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_5/ReadVariableOp_1?
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_5/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3?
activation_5/ReluRelu*batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@@2
activation_5/Relu?
max_unpooling2d_2/CastCast$max_pooling_with_argmax2d_1/Cast:y:0*

DstT0*

SrcT0*/
_output_shapes
:?????????@@@2
max_unpooling2d_2/Cast?
max_unpooling2d_2/ShapeShapeactivation_5/Relu:activations:0*
T0*
_output_shapes
:2
max_unpooling2d_2/Shape?
%max_unpooling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%max_unpooling2d_2/strided_slice/stack?
'max_unpooling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d_2/strided_slice/stack_1?
'max_unpooling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d_2/strided_slice/stack_2?
max_unpooling2d_2/strided_sliceStridedSlice max_unpooling2d_2/Shape:output:0.max_unpooling2d_2/strided_slice/stack:output:00max_unpooling2d_2/strided_slice/stack_1:output:00max_unpooling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
max_unpooling2d_2/strided_slice?
'max_unpooling2d_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d_2/strided_slice_1/stack?
)max_unpooling2d_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_2/strided_slice_1/stack_1?
)max_unpooling2d_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_2/strided_slice_1/stack_2?
!max_unpooling2d_2/strided_slice_1StridedSlice max_unpooling2d_2/Shape:output:00max_unpooling2d_2/strided_slice_1/stack:output:02max_unpooling2d_2/strided_slice_1/stack_1:output:02max_unpooling2d_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!max_unpooling2d_2/strided_slice_1t
max_unpooling2d_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d_2/mul/y?
max_unpooling2d_2/mulMul*max_unpooling2d_2/strided_slice_1:output:0 max_unpooling2d_2/mul/y:output:0*
T0*
_output_shapes
: 2
max_unpooling2d_2/mul?
'max_unpooling2d_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d_2/strided_slice_2/stack?
)max_unpooling2d_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_2/strided_slice_2/stack_1?
)max_unpooling2d_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_2/strided_slice_2/stack_2?
!max_unpooling2d_2/strided_slice_2StridedSlice max_unpooling2d_2/Shape:output:00max_unpooling2d_2/strided_slice_2/stack:output:02max_unpooling2d_2/strided_slice_2/stack_1:output:02max_unpooling2d_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!max_unpooling2d_2/strided_slice_2x
max_unpooling2d_2/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d_2/mul_1/y?
max_unpooling2d_2/mul_1Mul*max_unpooling2d_2/strided_slice_2:output:0"max_unpooling2d_2/mul_1/y:output:0*
T0*
_output_shapes
: 2
max_unpooling2d_2/mul_1?
'max_unpooling2d_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d_2/strided_slice_3/stack?
)max_unpooling2d_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_2/strided_slice_3/stack_1?
)max_unpooling2d_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_2/strided_slice_3/stack_2?
!max_unpooling2d_2/strided_slice_3StridedSlice max_unpooling2d_2/Shape:output:00max_unpooling2d_2/strided_slice_3/stack:output:02max_unpooling2d_2/strided_slice_3/stack_1:output:02max_unpooling2d_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!max_unpooling2d_2/strided_slice_3?
max_unpooling2d_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2!
max_unpooling2d_2/Reshape/shape?
max_unpooling2d_2/ReshapeReshapemax_unpooling2d_2/Cast:y:0(max_unpooling2d_2/Reshape/shape:output:0*
T0*#
_output_shapes
:?????????2
max_unpooling2d_2/Reshape?
 max_unpooling2d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 max_unpooling2d_2/ExpandDims/dim?
max_unpooling2d_2/ExpandDims
ExpandDims"max_unpooling2d_2/Reshape:output:0)max_unpooling2d_2/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2
max_unpooling2d_2/ExpandDims?
!max_unpooling2d_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2#
!max_unpooling2d_2/Reshape_1/shape?
max_unpooling2d_2/Reshape_1Reshapeactivation_5/Relu:activations:0*max_unpooling2d_2/Reshape_1/shape:output:0*
T0*#
_output_shapes
:?????????2
max_unpooling2d_2/Reshape_1?
max_unpooling2d_2/Rank/packedPack(max_unpooling2d_2/strided_slice:output:0max_unpooling2d_2/mul:z:0max_unpooling2d_2/mul_1:z:0*max_unpooling2d_2/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2
max_unpooling2d_2/Rank/packedr
max_unpooling2d_2/RankConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d_2/Rank?
max_unpooling2d_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
max_unpooling2d_2/range/start?
max_unpooling2d_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d_2/range/delta?
max_unpooling2d_2/rangeRange&max_unpooling2d_2/range/start:output:0max_unpooling2d_2/Rank:output:0&max_unpooling2d_2/range/delta:output:0*
_output_shapes
:2
max_unpooling2d_2/range?
max_unpooling2d_2/Prod/inputPack(max_unpooling2d_2/strided_slice:output:0max_unpooling2d_2/mul:z:0max_unpooling2d_2/mul_1:z:0*max_unpooling2d_2/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2
max_unpooling2d_2/Prod/input?
max_unpooling2d_2/ProdProd%max_unpooling2d_2/Prod/input:output:0 max_unpooling2d_2/range:output:0*
T0*
_output_shapes
: 2
max_unpooling2d_2/Prod?
!max_unpooling2d_2/ScatterNd/shapePackmax_unpooling2d_2/Prod:output:0*
N*
T0*
_output_shapes
:2#
!max_unpooling2d_2/ScatterNd/shape?
max_unpooling2d_2/ScatterNd	ScatterNd%max_unpooling2d_2/ExpandDims:output:0$max_unpooling2d_2/Reshape_1:output:0*max_unpooling2d_2/ScatterNd/shape:output:0*
T0*
Tindices0*#
_output_shapes
:?????????2
max_unpooling2d_2/ScatterNd?
!max_unpooling2d_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2#
!max_unpooling2d_2/Reshape_2/shape?
max_unpooling2d_2/Reshape_2Reshape$max_unpooling2d_2/ScatterNd:output:0*max_unpooling2d_2/Reshape_2/shape:output:0*
T0*1
_output_shapes
:???????????@2
max_unpooling2d_2/Reshape_2?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2D$max_unpooling2d_2/Reshape_2:output:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_6/BiasAdd?
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_6/ReadVariableOp?
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_6/ReadVariableOp_1?
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_6/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3?
activation_6/ReluRelu*batch_normalization_6/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
activation_6/Relu?
max_unpooling2d_3/CastCast"max_pooling_with_argmax2d/Cast:y:0*

DstT0*

SrcT0*1
_output_shapes
:??????????? 2
max_unpooling2d_3/Cast?
max_unpooling2d_3/ShapeShapeactivation_6/Relu:activations:0*
T0*
_output_shapes
:2
max_unpooling2d_3/Shape?
%max_unpooling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%max_unpooling2d_3/strided_slice/stack?
'max_unpooling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d_3/strided_slice/stack_1?
'max_unpooling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d_3/strided_slice/stack_2?
max_unpooling2d_3/strided_sliceStridedSlice max_unpooling2d_3/Shape:output:0.max_unpooling2d_3/strided_slice/stack:output:00max_unpooling2d_3/strided_slice/stack_1:output:00max_unpooling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
max_unpooling2d_3/strided_slice?
'max_unpooling2d_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d_3/strided_slice_1/stack?
)max_unpooling2d_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_3/strided_slice_1/stack_1?
)max_unpooling2d_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_3/strided_slice_1/stack_2?
!max_unpooling2d_3/strided_slice_1StridedSlice max_unpooling2d_3/Shape:output:00max_unpooling2d_3/strided_slice_1/stack:output:02max_unpooling2d_3/strided_slice_1/stack_1:output:02max_unpooling2d_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!max_unpooling2d_3/strided_slice_1t
max_unpooling2d_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d_3/mul/y?
max_unpooling2d_3/mulMul*max_unpooling2d_3/strided_slice_1:output:0 max_unpooling2d_3/mul/y:output:0*
T0*
_output_shapes
: 2
max_unpooling2d_3/mul?
'max_unpooling2d_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d_3/strided_slice_2/stack?
)max_unpooling2d_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_3/strided_slice_2/stack_1?
)max_unpooling2d_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_3/strided_slice_2/stack_2?
!max_unpooling2d_3/strided_slice_2StridedSlice max_unpooling2d_3/Shape:output:00max_unpooling2d_3/strided_slice_2/stack:output:02max_unpooling2d_3/strided_slice_2/stack_1:output:02max_unpooling2d_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!max_unpooling2d_3/strided_slice_2x
max_unpooling2d_3/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d_3/mul_1/y?
max_unpooling2d_3/mul_1Mul*max_unpooling2d_3/strided_slice_2:output:0"max_unpooling2d_3/mul_1/y:output:0*
T0*
_output_shapes
: 2
max_unpooling2d_3/mul_1?
'max_unpooling2d_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d_3/strided_slice_3/stack?
)max_unpooling2d_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_3/strided_slice_3/stack_1?
)max_unpooling2d_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_3/strided_slice_3/stack_2?
!max_unpooling2d_3/strided_slice_3StridedSlice max_unpooling2d_3/Shape:output:00max_unpooling2d_3/strided_slice_3/stack:output:02max_unpooling2d_3/strided_slice_3/stack_1:output:02max_unpooling2d_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!max_unpooling2d_3/strided_slice_3?
max_unpooling2d_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2!
max_unpooling2d_3/Reshape/shape?
max_unpooling2d_3/ReshapeReshapemax_unpooling2d_3/Cast:y:0(max_unpooling2d_3/Reshape/shape:output:0*
T0*#
_output_shapes
:?????????2
max_unpooling2d_3/Reshape?
 max_unpooling2d_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 max_unpooling2d_3/ExpandDims/dim?
max_unpooling2d_3/ExpandDims
ExpandDims"max_unpooling2d_3/Reshape:output:0)max_unpooling2d_3/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2
max_unpooling2d_3/ExpandDims?
!max_unpooling2d_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2#
!max_unpooling2d_3/Reshape_1/shape?
max_unpooling2d_3/Reshape_1Reshapeactivation_6/Relu:activations:0*max_unpooling2d_3/Reshape_1/shape:output:0*
T0*#
_output_shapes
:?????????2
max_unpooling2d_3/Reshape_1?
max_unpooling2d_3/Rank/packedPack(max_unpooling2d_3/strided_slice:output:0max_unpooling2d_3/mul:z:0max_unpooling2d_3/mul_1:z:0*max_unpooling2d_3/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2
max_unpooling2d_3/Rank/packedr
max_unpooling2d_3/RankConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d_3/Rank?
max_unpooling2d_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
max_unpooling2d_3/range/start?
max_unpooling2d_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d_3/range/delta?
max_unpooling2d_3/rangeRange&max_unpooling2d_3/range/start:output:0max_unpooling2d_3/Rank:output:0&max_unpooling2d_3/range/delta:output:0*
_output_shapes
:2
max_unpooling2d_3/range?
max_unpooling2d_3/Prod/inputPack(max_unpooling2d_3/strided_slice:output:0max_unpooling2d_3/mul:z:0max_unpooling2d_3/mul_1:z:0*max_unpooling2d_3/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2
max_unpooling2d_3/Prod/input?
max_unpooling2d_3/ProdProd%max_unpooling2d_3/Prod/input:output:0 max_unpooling2d_3/range:output:0*
T0*
_output_shapes
: 2
max_unpooling2d_3/Prod?
!max_unpooling2d_3/ScatterNd/shapePackmax_unpooling2d_3/Prod:output:0*
N*
T0*
_output_shapes
:2#
!max_unpooling2d_3/ScatterNd/shape?
max_unpooling2d_3/ScatterNd	ScatterNd%max_unpooling2d_3/ExpandDims:output:0$max_unpooling2d_3/Reshape_1:output:0*max_unpooling2d_3/ScatterNd/shape:output:0*
T0*
Tindices0*#
_output_shapes
:?????????2
max_unpooling2d_3/ScatterNd?
!max_unpooling2d_3/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????          2#
!max_unpooling2d_3/Reshape_2/shape?
max_unpooling2d_3/Reshape_2Reshape$max_unpooling2d_3/ScatterNd:output:0*max_unpooling2d_3/Reshape_2/shape:output:0*
T0*1
_output_shapes
:??????????? 2
max_unpooling2d_3/Reshape_2?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_7/Conv2D/ReadVariableOp?
conv2d_7/Conv2DConv2D$max_unpooling2d_3/Reshape_2:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_7/Conv2D?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_7/BiasAdd?
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_7/ReadVariableOp?
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_7/ReadVariableOp_1?
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_7/BiasAdd:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3?
activation_7/SigmoidSigmoid*batch_normalization_7/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
activation_7/Sigmoid}
IdentityIdentityactivation_7/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes
}:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_3_layer_call_fn_10050

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_57522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
(__inference_conv2d_4_layer_call_fn_10198

inputs"
unknown:?@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_66672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????  ?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9635

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
c
G__inference_activation_4_layer_call_and_return_conditional_losses_10327

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????  @2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
U__inference_max_pooling_with_argmax2d_2_layer_call_and_return_conditional_losses_7464

inputs
identity

identity_1?
MaxPoolWithArgmaxMaxPoolWithArgmaxinputs*
T0*J
_output_shapes8
6:?????????  @:?????????  @*
ksize
*
paddingSAME*
strides
2
MaxPoolWithArgmaxy
CastCastMaxPoolWithArgmax:argmax:0*

DstT0*

SrcT0	*/
_output_shapes
:?????????  @2
Castv
IdentityIdentityMaxPoolWithArgmax:output:0*
T0*/
_output_shapes
:?????????  @2

Identityh

Identity_1IdentityCast:y:0*
T0*/
_output_shapes
:?????????  @2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@@:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_10455

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_5_layer_call_fn_10499

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_60482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
`
D__inference_activation_layer_call_and_return_conditional_losses_6415

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:??????????? 2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
d
8__inference_max_pooling_with_argmax2d_layer_call_fn_9573

inputs
identity

identity_1?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::??????????? :??????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_max_pooling_with_argmax2d_layer_call_and_return_conditional_losses_64252
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:??????????? 2

Identityz

Identity_1IdentityPartitionedCall:output:1*
T0*1
_output_shapes
:??????????? 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
??
?
?__inference_model_layer_call_and_return_conditional_losses_7934

inputs%
conv2d_7801: 
conv2d_7803: &
batch_normalization_7806: &
batch_normalization_7808: &
batch_normalization_7810: &
batch_normalization_7812: '
conv2d_1_7818: @
conv2d_1_7820:@(
batch_normalization_1_7823:@(
batch_normalization_1_7825:@(
batch_normalization_1_7827:@(
batch_normalization_1_7829:@'
conv2d_2_7835:@@
conv2d_2_7837:@(
batch_normalization_2_7840:@(
batch_normalization_2_7842:@(
batch_normalization_2_7844:@(
batch_normalization_2_7846:@(
conv2d_3_7852:@?
conv2d_3_7854:	?)
batch_normalization_3_7857:	?)
batch_normalization_3_7859:	?)
batch_normalization_3_7861:	?)
batch_normalization_3_7863:	?(
conv2d_4_7870:?@
conv2d_4_7872:@(
batch_normalization_4_7875:@(
batch_normalization_4_7877:@(
batch_normalization_4_7879:@(
batch_normalization_4_7881:@'
conv2d_5_7886:@@
conv2d_5_7888:@(
batch_normalization_5_7891:@(
batch_normalization_5_7893:@(
batch_normalization_5_7895:@(
batch_normalization_5_7897:@'
conv2d_6_7902:@ 
conv2d_6_7904: (
batch_normalization_6_7907: (
batch_normalization_6_7909: (
batch_normalization_6_7911: (
batch_normalization_6_7913: '
conv2d_7_7918: 
conv2d_7_7920:(
batch_normalization_7_7923:(
batch_normalization_7_7925:(
batch_normalization_7_7927:(
batch_normalization_7_7929:
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?-batch_normalization_5/StatefulPartitionedCall?-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_7801conv2d_7803*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_63772 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_7806batch_normalization_7808batch_normalization_7810batch_normalization_7812*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_76742-
+batch_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_64152
activation/PartitionedCall?
)max_pooling_with_argmax2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::??????????? :??????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_max_pooling_with_argmax2d_layer_call_and_return_conditional_losses_76302+
)max_pooling_with_argmax2d/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall2max_pooling_with_argmax2d/PartitionedCall:output:0conv2d_1_7818conv2d_1_7820*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_64382"
 conv2d_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_7823batch_normalization_1_7825batch_normalization_1_7827batch_normalization_1_7829*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_75912/
-batch_normalization_1/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_64762
activation_1/PartitionedCall?
+max_pooling_with_argmax2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:?????????@@@:?????????@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_max_pooling_with_argmax2d_1_layer_call_and_return_conditional_losses_75472-
+max_pooling_with_argmax2d_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall4max_pooling_with_argmax2d_1/PartitionedCall:output:0conv2d_2_7835conv2d_2_7837*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_64992"
 conv2d_2/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_7840batch_normalization_2_7842batch_normalization_2_7844batch_normalization_2_7846*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_75082/
-batch_normalization_2/StatefulPartitionedCall?
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_65372
activation_2/PartitionedCall?
+max_pooling_with_argmax2d_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:?????????  @:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_max_pooling_with_argmax2d_2_layer_call_and_return_conditional_losses_74642-
+max_pooling_with_argmax2d_2/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall4max_pooling_with_argmax2d_2/PartitionedCall:output:0conv2d_3_7852conv2d_3_7854*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_65602"
 conv2d_3/StatefulPartitionedCall?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_7857batch_normalization_3_7859batch_normalization_3_7861batch_normalization_3_7863*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_74252/
-batch_normalization_3/StatefulPartitionedCall?
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_65982
activation_3/PartitionedCall?
+max_pooling_with_argmax2d_3/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *L
_output_shapes:
8:??????????:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_max_pooling_with_argmax2d_3_layer_call_and_return_conditional_losses_73812-
+max_pooling_with_argmax2d_3/PartitionedCall?
max_unpooling2d/PartitionedCallPartitionedCall4max_pooling_with_argmax2d_3/PartitionedCall:output:04max_pooling_with_argmax2d_3/PartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_unpooling2d_layer_call_and_return_conditional_losses_66552!
max_unpooling2d/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_unpooling2d/PartitionedCall:output:0conv2d_4_7870conv2d_4_7872*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_66672"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_7875batch_normalization_4_7877batch_normalization_4_7879batch_normalization_4_7881*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_73352/
-batch_normalization_4/StatefulPartitionedCall?
activation_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_67052
activation_4/PartitionedCall?
!max_unpooling2d_1/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:04max_pooling_with_argmax2d_2/PartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_unpooling2d_1_layer_call_and_return_conditional_losses_67512#
!max_unpooling2d_1/PartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall*max_unpooling2d_1/PartitionedCall:output:0conv2d_5_7886conv2d_5_7888*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_67632"
 conv2d_5/StatefulPartitionedCall?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_7891batch_normalization_5_7893batch_normalization_5_7895batch_normalization_5_7897*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_72682/
-batch_normalization_5/StatefulPartitionedCall?
activation_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_5_layer_call_and_return_conditional_losses_68012
activation_5/PartitionedCall?
!max_unpooling2d_2/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:04max_pooling_with_argmax2d_1/PartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_unpooling2d_2_layer_call_and_return_conditional_losses_68472#
!max_unpooling2d_2/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall*max_unpooling2d_2/PartitionedCall:output:0conv2d_6_7902conv2d_6_7904*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_68592"
 conv2d_6/StatefulPartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_7907batch_normalization_6_7909batch_normalization_6_7911batch_normalization_6_7913*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_72012/
-batch_normalization_6/StatefulPartitionedCall?
activation_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_6_layer_call_and_return_conditional_losses_68972
activation_6/PartitionedCall?
!max_unpooling2d_3/PartitionedCallPartitionedCall%activation_6/PartitionedCall:output:02max_pooling_with_argmax2d/PartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_unpooling2d_3_layer_call_and_return_conditional_losses_69432#
!max_unpooling2d_3/PartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall*max_unpooling2d_3/PartitionedCall:output:0conv2d_7_7918conv2d_7_7920*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_7_layer_call_and_return_conditional_losses_69552"
 conv2d_7/StatefulPartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_7_7923batch_normalization_7_7925batch_normalization_7_7927batch_normalization_7_7929*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_71342/
-batch_normalization_7/StatefulPartitionedCall?
activation_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_7_layer_call_and_return_conditional_losses_69932
activation_7/PartitionedCall?
IdentityIdentity%activation_7/PartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes
}:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_5_layer_call_fn_10486

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_60042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_10437

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
B__inference_conv2d_4_layer_call_and_return_conditional_losses_6667

inputs9
conv2d_readvariableop_resource:?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????  @2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????  ?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
U__inference_max_pooling_with_argmax2d_2_layer_call_and_return_conditional_losses_9924

inputs
identity

identity_1?
MaxPoolWithArgmaxMaxPoolWithArgmaxinputs*
T0*J
_output_shapes8
6:?????????  @:?????????  @*
ksize
*
paddingSAME*
strides
2
MaxPoolWithArgmaxy
CastCastMaxPoolWithArgmax:argmax:0*

DstT0*

SrcT0	*/
_output_shapes
:?????????  @2
Castv
IdentityIdentityMaxPoolWithArgmax:output:0*
T0*/
_output_shapes
:?????????  @2

Identityh

Identity_1IdentityCast:y:0*
T0*/
_output_shapes
:?????????  @2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@@:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
5__inference_batch_normalization_5_layer_call_fn_10512

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_67862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_5418

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_10189

inputs9
conv2d_readvariableop_resource:?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????  @2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????  ?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
g
;__inference_max_pooling_with_argmax2d_3_layer_call_fn_10129

inputs
identity

identity_1?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *L
_output_shapes:
8:??????????:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_max_pooling_with_argmax2d_3_layer_call_and_return_conditional_losses_73812
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identityy

Identity_1IdentityPartitionedCall:output:1*
T0*0
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
ݍ
?.
__inference__wrapped_model_5352
input_1E
+model_conv2d_conv2d_readvariableop_resource: :
,model_conv2d_biasadd_readvariableop_resource: ?
1model_batch_normalization_readvariableop_resource: A
3model_batch_normalization_readvariableop_1_resource: P
Bmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource: R
Dmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource: G
-model_conv2d_1_conv2d_readvariableop_resource: @<
.model_conv2d_1_biasadd_readvariableop_resource:@A
3model_batch_normalization_1_readvariableop_resource:@C
5model_batch_normalization_1_readvariableop_1_resource:@R
Dmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@T
Fmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@G
-model_conv2d_2_conv2d_readvariableop_resource:@@<
.model_conv2d_2_biasadd_readvariableop_resource:@A
3model_batch_normalization_2_readvariableop_resource:@C
5model_batch_normalization_2_readvariableop_1_resource:@R
Dmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@T
Fmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@H
-model_conv2d_3_conv2d_readvariableop_resource:@?=
.model_conv2d_3_biasadd_readvariableop_resource:	?B
3model_batch_normalization_3_readvariableop_resource:	?D
5model_batch_normalization_3_readvariableop_1_resource:	?S
Dmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	?U
Fmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	?H
-model_conv2d_4_conv2d_readvariableop_resource:?@<
.model_conv2d_4_biasadd_readvariableop_resource:@A
3model_batch_normalization_4_readvariableop_resource:@C
5model_batch_normalization_4_readvariableop_1_resource:@R
Dmodel_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:@T
Fmodel_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:@G
-model_conv2d_5_conv2d_readvariableop_resource:@@<
.model_conv2d_5_biasadd_readvariableop_resource:@A
3model_batch_normalization_5_readvariableop_resource:@C
5model_batch_normalization_5_readvariableop_1_resource:@R
Dmodel_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@T
Fmodel_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@G
-model_conv2d_6_conv2d_readvariableop_resource:@ <
.model_conv2d_6_biasadd_readvariableop_resource: A
3model_batch_normalization_6_readvariableop_resource: C
5model_batch_normalization_6_readvariableop_1_resource: R
Dmodel_batch_normalization_6_fusedbatchnormv3_readvariableop_resource: T
Fmodel_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource: G
-model_conv2d_7_conv2d_readvariableop_resource: <
.model_conv2d_7_biasadd_readvariableop_resource:A
3model_batch_normalization_7_readvariableop_resource:C
5model_batch_normalization_7_readvariableop_1_resource:R
Dmodel_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:T
Fmodel_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:
identity??9model/batch_normalization/FusedBatchNormV3/ReadVariableOp?;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?(model/batch_normalization/ReadVariableOp?*model/batch_normalization/ReadVariableOp_1?;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?*model/batch_normalization_1/ReadVariableOp?,model/batch_normalization_1/ReadVariableOp_1?;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?*model/batch_normalization_2/ReadVariableOp?,model/batch_normalization_2/ReadVariableOp_1?;model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?=model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?*model/batch_normalization_3/ReadVariableOp?,model/batch_normalization_3/ReadVariableOp_1?;model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?=model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?*model/batch_normalization_4/ReadVariableOp?,model/batch_normalization_4/ReadVariableOp_1?;model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?=model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?*model/batch_normalization_5/ReadVariableOp?,model/batch_normalization_5/ReadVariableOp_1?;model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?=model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?*model/batch_normalization_6/ReadVariableOp?,model/batch_normalization_6/ReadVariableOp_1?;model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?=model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?*model/batch_normalization_7/ReadVariableOp?,model/batch_normalization_7/ReadVariableOp_1?#model/conv2d/BiasAdd/ReadVariableOp?"model/conv2d/Conv2D/ReadVariableOp?%model/conv2d_1/BiasAdd/ReadVariableOp?$model/conv2d_1/Conv2D/ReadVariableOp?%model/conv2d_2/BiasAdd/ReadVariableOp?$model/conv2d_2/Conv2D/ReadVariableOp?%model/conv2d_3/BiasAdd/ReadVariableOp?$model/conv2d_3/Conv2D/ReadVariableOp?%model/conv2d_4/BiasAdd/ReadVariableOp?$model/conv2d_4/Conv2D/ReadVariableOp?%model/conv2d_5/BiasAdd/ReadVariableOp?$model/conv2d_5/Conv2D/ReadVariableOp?%model/conv2d_6/BiasAdd/ReadVariableOp?$model/conv2d_6/Conv2D/ReadVariableOp?%model/conv2d_7/BiasAdd/ReadVariableOp?$model/conv2d_7/Conv2D/ReadVariableOp?
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02$
"model/conv2d/Conv2D/ReadVariableOp?
model/conv2d/Conv2DConv2Dinput_1*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
model/conv2d/Conv2D?
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#model/conv2d/BiasAdd/ReadVariableOp?
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
model/conv2d/BiasAdd?
(model/batch_normalization/ReadVariableOpReadVariableOp1model_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02*
(model/batch_normalization/ReadVariableOp?
*model/batch_normalization/ReadVariableOp_1ReadVariableOp3model_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02,
*model/batch_normalization/ReadVariableOp_1?
9model/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpBmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02;
9model/batch_normalization/FusedBatchNormV3/ReadVariableOp?
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02=
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
*model/batch_normalization/FusedBatchNormV3FusedBatchNormV3model/conv2d/BiasAdd:output:00model/batch_normalization/ReadVariableOp:value:02model/batch_normalization/ReadVariableOp_1:value:0Amodel/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Cmodel/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2,
*model/batch_normalization/FusedBatchNormV3?
model/activation/ReluRelu.model/batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
model/activation/Relu?
1model/max_pooling_with_argmax2d/MaxPoolWithArgmaxMaxPoolWithArgmax#model/activation/Relu:activations:0*
T0*N
_output_shapes<
::??????????? :??????????? *
ksize
*
paddingSAME*
strides
23
1model/max_pooling_with_argmax2d/MaxPoolWithArgmax?
$model/max_pooling_with_argmax2d/CastCast:model/max_pooling_with_argmax2d/MaxPoolWithArgmax:argmax:0*

DstT0*

SrcT0	*1
_output_shapes
:??????????? 2&
$model/max_pooling_with_argmax2d/Cast?
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02&
$model/conv2d_1/Conv2D/ReadVariableOp?
model/conv2d_1/Conv2DConv2D:model/max_pooling_with_argmax2d/MaxPoolWithArgmax:output:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
model/conv2d_1/Conv2D?
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model/conv2d_1/BiasAdd/ReadVariableOp?
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
model/conv2d_1/BiasAdd?
*model/batch_normalization_1/ReadVariableOpReadVariableOp3model_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype02,
*model/batch_normalization_1/ReadVariableOp?
,model/batch_normalization_1/ReadVariableOp_1ReadVariableOp5model_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02.
,model/batch_normalization_1/ReadVariableOp_1?
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02=
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02?
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
,model/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3model/conv2d_1/BiasAdd:output:02model/batch_normalization_1/ReadVariableOp:value:04model/batch_normalization_1/ReadVariableOp_1:value:0Cmodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2.
,model/batch_normalization_1/FusedBatchNormV3?
model/activation_1/ReluRelu0model/batch_normalization_1/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????@2
model/activation_1/Relu?
3model/max_pooling_with_argmax2d_1/MaxPoolWithArgmaxMaxPoolWithArgmax%model/activation_1/Relu:activations:0*
T0*J
_output_shapes8
6:?????????@@@:?????????@@@*
ksize
*
paddingSAME*
strides
25
3model/max_pooling_with_argmax2d_1/MaxPoolWithArgmax?
&model/max_pooling_with_argmax2d_1/CastCast<model/max_pooling_with_argmax2d_1/MaxPoolWithArgmax:argmax:0*

DstT0*

SrcT0	*/
_output_shapes
:?????????@@@2(
&model/max_pooling_with_argmax2d_1/Cast?
$model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02&
$model/conv2d_2/Conv2D/ReadVariableOp?
model/conv2d_2/Conv2DConv2D<model/max_pooling_with_argmax2d_1/MaxPoolWithArgmax:output:0,model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
2
model/conv2d_2/Conv2D?
%model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model/conv2d_2/BiasAdd/ReadVariableOp?
model/conv2d_2/BiasAddBiasAddmodel/conv2d_2/Conv2D:output:0-model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@2
model/conv2d_2/BiasAdd?
*model/batch_normalization_2/ReadVariableOpReadVariableOp3model_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02,
*model/batch_normalization_2/ReadVariableOp?
,model/batch_normalization_2/ReadVariableOp_1ReadVariableOp5model_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02.
,model/batch_normalization_2/ReadVariableOp_1?
;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02=
;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02?
=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
,model/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3model/conv2d_2/BiasAdd:output:02model/batch_normalization_2/ReadVariableOp:value:04model/batch_normalization_2/ReadVariableOp_1:value:0Cmodel/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
is_training( 2.
,model/batch_normalization_2/FusedBatchNormV3?
model/activation_2/ReluRelu0model/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@@2
model/activation_2/Relu?
3model/max_pooling_with_argmax2d_2/MaxPoolWithArgmaxMaxPoolWithArgmax%model/activation_2/Relu:activations:0*
T0*J
_output_shapes8
6:?????????  @:?????????  @*
ksize
*
paddingSAME*
strides
25
3model/max_pooling_with_argmax2d_2/MaxPoolWithArgmax?
&model/max_pooling_with_argmax2d_2/CastCast<model/max_pooling_with_argmax2d_2/MaxPoolWithArgmax:argmax:0*

DstT0*

SrcT0	*/
_output_shapes
:?????????  @2(
&model/max_pooling_with_argmax2d_2/Cast?
$model/conv2d_3/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02&
$model/conv2d_3/Conv2D/ReadVariableOp?
model/conv2d_3/Conv2DConv2D<model/max_pooling_with_argmax2d_2/MaxPoolWithArgmax:output:0,model/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2
model/conv2d_3/Conv2D?
%model/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%model/conv2d_3/BiasAdd/ReadVariableOp?
model/conv2d_3/BiasAddBiasAddmodel/conv2d_3/Conv2D:output:0-model/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?2
model/conv2d_3/BiasAdd?
*model/batch_normalization_3/ReadVariableOpReadVariableOp3model_batch_normalization_3_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*model/batch_normalization_3/ReadVariableOp?
,model/batch_normalization_3/ReadVariableOp_1ReadVariableOp5model_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02.
,model/batch_normalization_3/ReadVariableOp_1?
;model/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02=
;model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
=model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02?
=model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
,model/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3model/conv2d_3/BiasAdd:output:02model/batch_normalization_3/ReadVariableOp:value:04model/batch_normalization_3/ReadVariableOp_1:value:0Cmodel/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
is_training( 2.
,model/batch_normalization_3/FusedBatchNormV3?
model/activation_3/ReluRelu0model/batch_normalization_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????  ?2
model/activation_3/Relu?
3model/max_pooling_with_argmax2d_3/MaxPoolWithArgmaxMaxPoolWithArgmax%model/activation_3/Relu:activations:0*
T0*L
_output_shapes:
8:??????????:??????????*
ksize
*
paddingSAME*
strides
25
3model/max_pooling_with_argmax2d_3/MaxPoolWithArgmax?
&model/max_pooling_with_argmax2d_3/CastCast<model/max_pooling_with_argmax2d_3/MaxPoolWithArgmax:argmax:0*

DstT0*

SrcT0	*0
_output_shapes
:??????????2(
&model/max_pooling_with_argmax2d_3/Cast?
model/max_unpooling2d/CastCast*model/max_pooling_with_argmax2d_3/Cast:y:0*

DstT0*

SrcT0*0
_output_shapes
:??????????2
model/max_unpooling2d/Cast?
model/max_unpooling2d/ShapeShape<model/max_pooling_with_argmax2d_3/MaxPoolWithArgmax:output:0*
T0*
_output_shapes
:2
model/max_unpooling2d/Shape?
)model/max_unpooling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)model/max_unpooling2d/strided_slice/stack?
+model/max_unpooling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+model/max_unpooling2d/strided_slice/stack_1?
+model/max_unpooling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+model/max_unpooling2d/strided_slice/stack_2?
#model/max_unpooling2d/strided_sliceStridedSlice$model/max_unpooling2d/Shape:output:02model/max_unpooling2d/strided_slice/stack:output:04model/max_unpooling2d/strided_slice/stack_1:output:04model/max_unpooling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#model/max_unpooling2d/strided_slice?
+model/max_unpooling2d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+model/max_unpooling2d/strided_slice_1/stack?
-model/max_unpooling2d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-model/max_unpooling2d/strided_slice_1/stack_1?
-model/max_unpooling2d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-model/max_unpooling2d/strided_slice_1/stack_2?
%model/max_unpooling2d/strided_slice_1StridedSlice$model/max_unpooling2d/Shape:output:04model/max_unpooling2d/strided_slice_1/stack:output:06model/max_unpooling2d/strided_slice_1/stack_1:output:06model/max_unpooling2d/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%model/max_unpooling2d/strided_slice_1|
model/max_unpooling2d/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
model/max_unpooling2d/mul/y?
model/max_unpooling2d/mulMul.model/max_unpooling2d/strided_slice_1:output:0$model/max_unpooling2d/mul/y:output:0*
T0*
_output_shapes
: 2
model/max_unpooling2d/mul?
+model/max_unpooling2d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+model/max_unpooling2d/strided_slice_2/stack?
-model/max_unpooling2d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-model/max_unpooling2d/strided_slice_2/stack_1?
-model/max_unpooling2d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-model/max_unpooling2d/strided_slice_2/stack_2?
%model/max_unpooling2d/strided_slice_2StridedSlice$model/max_unpooling2d/Shape:output:04model/max_unpooling2d/strided_slice_2/stack:output:06model/max_unpooling2d/strided_slice_2/stack_1:output:06model/max_unpooling2d/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%model/max_unpooling2d/strided_slice_2?
model/max_unpooling2d/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
model/max_unpooling2d/mul_1/y?
model/max_unpooling2d/mul_1Mul.model/max_unpooling2d/strided_slice_2:output:0&model/max_unpooling2d/mul_1/y:output:0*
T0*
_output_shapes
: 2
model/max_unpooling2d/mul_1?
+model/max_unpooling2d/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+model/max_unpooling2d/strided_slice_3/stack?
-model/max_unpooling2d/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-model/max_unpooling2d/strided_slice_3/stack_1?
-model/max_unpooling2d/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-model/max_unpooling2d/strided_slice_3/stack_2?
%model/max_unpooling2d/strided_slice_3StridedSlice$model/max_unpooling2d/Shape:output:04model/max_unpooling2d/strided_slice_3/stack:output:06model/max_unpooling2d/strided_slice_3/stack_1:output:06model/max_unpooling2d/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%model/max_unpooling2d/strided_slice_3?
#model/max_unpooling2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2%
#model/max_unpooling2d/Reshape/shape?
model/max_unpooling2d/ReshapeReshapemodel/max_unpooling2d/Cast:y:0,model/max_unpooling2d/Reshape/shape:output:0*
T0*#
_output_shapes
:?????????2
model/max_unpooling2d/Reshape?
$model/max_unpooling2d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$model/max_unpooling2d/ExpandDims/dim?
 model/max_unpooling2d/ExpandDims
ExpandDims&model/max_unpooling2d/Reshape:output:0-model/max_unpooling2d/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2"
 model/max_unpooling2d/ExpandDims?
%model/max_unpooling2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2'
%model/max_unpooling2d/Reshape_1/shape?
model/max_unpooling2d/Reshape_1Reshape<model/max_pooling_with_argmax2d_3/MaxPoolWithArgmax:output:0.model/max_unpooling2d/Reshape_1/shape:output:0*
T0*#
_output_shapes
:?????????2!
model/max_unpooling2d/Reshape_1?
!model/max_unpooling2d/Rank/packedPack,model/max_unpooling2d/strided_slice:output:0model/max_unpooling2d/mul:z:0model/max_unpooling2d/mul_1:z:0.model/max_unpooling2d/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2#
!model/max_unpooling2d/Rank/packedz
model/max_unpooling2d/RankConst*
_output_shapes
: *
dtype0*
value	B :2
model/max_unpooling2d/Rank?
!model/max_unpooling2d/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2#
!model/max_unpooling2d/range/start?
!model/max_unpooling2d/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2#
!model/max_unpooling2d/range/delta?
model/max_unpooling2d/rangeRange*model/max_unpooling2d/range/start:output:0#model/max_unpooling2d/Rank:output:0*model/max_unpooling2d/range/delta:output:0*
_output_shapes
:2
model/max_unpooling2d/range?
 model/max_unpooling2d/Prod/inputPack,model/max_unpooling2d/strided_slice:output:0model/max_unpooling2d/mul:z:0model/max_unpooling2d/mul_1:z:0.model/max_unpooling2d/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2"
 model/max_unpooling2d/Prod/input?
model/max_unpooling2d/ProdProd)model/max_unpooling2d/Prod/input:output:0$model/max_unpooling2d/range:output:0*
T0*
_output_shapes
: 2
model/max_unpooling2d/Prod?
%model/max_unpooling2d/ScatterNd/shapePack#model/max_unpooling2d/Prod:output:0*
N*
T0*
_output_shapes
:2'
%model/max_unpooling2d/ScatterNd/shape?
model/max_unpooling2d/ScatterNd	ScatterNd)model/max_unpooling2d/ExpandDims:output:0(model/max_unpooling2d/Reshape_1:output:0.model/max_unpooling2d/ScatterNd/shape:output:0*
T0*
Tindices0*#
_output_shapes
:?????????2!
model/max_unpooling2d/ScatterNd?
%model/max_unpooling2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????        ?   2'
%model/max_unpooling2d/Reshape_2/shape?
model/max_unpooling2d/Reshape_2Reshape(model/max_unpooling2d/ScatterNd:output:0.model/max_unpooling2d/Reshape_2/shape:output:0*
T0*0
_output_shapes
:?????????  ?2!
model/max_unpooling2d/Reshape_2?
$model/conv2d_4/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02&
$model/conv2d_4/Conv2D/ReadVariableOp?
model/conv2d_4/Conv2DConv2D(model/max_unpooling2d/Reshape_2:output:0,model/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2
model/conv2d_4/Conv2D?
%model/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model/conv2d_4/BiasAdd/ReadVariableOp?
model/conv2d_4/BiasAddBiasAddmodel/conv2d_4/Conv2D:output:0-model/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2
model/conv2d_4/BiasAdd?
*model/batch_normalization_4/ReadVariableOpReadVariableOp3model_batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype02,
*model/batch_normalization_4/ReadVariableOp?
,model/batch_normalization_4/ReadVariableOp_1ReadVariableOp5model_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02.
,model/batch_normalization_4/ReadVariableOp_1?
;model/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02=
;model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
=model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02?
=model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
,model/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3model/conv2d_4/BiasAdd:output:02model/batch_normalization_4/ReadVariableOp:value:04model/batch_normalization_4/ReadVariableOp_1:value:0Cmodel/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( 2.
,model/batch_normalization_4/FusedBatchNormV3?
model/activation_4/ReluRelu0model/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  @2
model/activation_4/Relu?
model/max_unpooling2d_1/CastCast*model/max_pooling_with_argmax2d_2/Cast:y:0*

DstT0*

SrcT0*/
_output_shapes
:?????????  @2
model/max_unpooling2d_1/Cast?
model/max_unpooling2d_1/ShapeShape%model/activation_4/Relu:activations:0*
T0*
_output_shapes
:2
model/max_unpooling2d_1/Shape?
+model/max_unpooling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+model/max_unpooling2d_1/strided_slice/stack?
-model/max_unpooling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-model/max_unpooling2d_1/strided_slice/stack_1?
-model/max_unpooling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-model/max_unpooling2d_1/strided_slice/stack_2?
%model/max_unpooling2d_1/strided_sliceStridedSlice&model/max_unpooling2d_1/Shape:output:04model/max_unpooling2d_1/strided_slice/stack:output:06model/max_unpooling2d_1/strided_slice/stack_1:output:06model/max_unpooling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%model/max_unpooling2d_1/strided_slice?
-model/max_unpooling2d_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-model/max_unpooling2d_1/strided_slice_1/stack?
/model/max_unpooling2d_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/model/max_unpooling2d_1/strided_slice_1/stack_1?
/model/max_unpooling2d_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/model/max_unpooling2d_1/strided_slice_1/stack_2?
'model/max_unpooling2d_1/strided_slice_1StridedSlice&model/max_unpooling2d_1/Shape:output:06model/max_unpooling2d_1/strided_slice_1/stack:output:08model/max_unpooling2d_1/strided_slice_1/stack_1:output:08model/max_unpooling2d_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'model/max_unpooling2d_1/strided_slice_1?
model/max_unpooling2d_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
model/max_unpooling2d_1/mul/y?
model/max_unpooling2d_1/mulMul0model/max_unpooling2d_1/strided_slice_1:output:0&model/max_unpooling2d_1/mul/y:output:0*
T0*
_output_shapes
: 2
model/max_unpooling2d_1/mul?
-model/max_unpooling2d_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-model/max_unpooling2d_1/strided_slice_2/stack?
/model/max_unpooling2d_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/model/max_unpooling2d_1/strided_slice_2/stack_1?
/model/max_unpooling2d_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/model/max_unpooling2d_1/strided_slice_2/stack_2?
'model/max_unpooling2d_1/strided_slice_2StridedSlice&model/max_unpooling2d_1/Shape:output:06model/max_unpooling2d_1/strided_slice_2/stack:output:08model/max_unpooling2d_1/strided_slice_2/stack_1:output:08model/max_unpooling2d_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'model/max_unpooling2d_1/strided_slice_2?
model/max_unpooling2d_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
model/max_unpooling2d_1/mul_1/y?
model/max_unpooling2d_1/mul_1Mul0model/max_unpooling2d_1/strided_slice_2:output:0(model/max_unpooling2d_1/mul_1/y:output:0*
T0*
_output_shapes
: 2
model/max_unpooling2d_1/mul_1?
-model/max_unpooling2d_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-model/max_unpooling2d_1/strided_slice_3/stack?
/model/max_unpooling2d_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/model/max_unpooling2d_1/strided_slice_3/stack_1?
/model/max_unpooling2d_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/model/max_unpooling2d_1/strided_slice_3/stack_2?
'model/max_unpooling2d_1/strided_slice_3StridedSlice&model/max_unpooling2d_1/Shape:output:06model/max_unpooling2d_1/strided_slice_3/stack:output:08model/max_unpooling2d_1/strided_slice_3/stack_1:output:08model/max_unpooling2d_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'model/max_unpooling2d_1/strided_slice_3?
%model/max_unpooling2d_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2'
%model/max_unpooling2d_1/Reshape/shape?
model/max_unpooling2d_1/ReshapeReshape model/max_unpooling2d_1/Cast:y:0.model/max_unpooling2d_1/Reshape/shape:output:0*
T0*#
_output_shapes
:?????????2!
model/max_unpooling2d_1/Reshape?
&model/max_unpooling2d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&model/max_unpooling2d_1/ExpandDims/dim?
"model/max_unpooling2d_1/ExpandDims
ExpandDims(model/max_unpooling2d_1/Reshape:output:0/model/max_unpooling2d_1/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2$
"model/max_unpooling2d_1/ExpandDims?
'model/max_unpooling2d_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2)
'model/max_unpooling2d_1/Reshape_1/shape?
!model/max_unpooling2d_1/Reshape_1Reshape%model/activation_4/Relu:activations:00model/max_unpooling2d_1/Reshape_1/shape:output:0*
T0*#
_output_shapes
:?????????2#
!model/max_unpooling2d_1/Reshape_1?
#model/max_unpooling2d_1/Rank/packedPack.model/max_unpooling2d_1/strided_slice:output:0model/max_unpooling2d_1/mul:z:0!model/max_unpooling2d_1/mul_1:z:00model/max_unpooling2d_1/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2%
#model/max_unpooling2d_1/Rank/packed~
model/max_unpooling2d_1/RankConst*
_output_shapes
: *
dtype0*
value	B :2
model/max_unpooling2d_1/Rank?
#model/max_unpooling2d_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2%
#model/max_unpooling2d_1/range/start?
#model/max_unpooling2d_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2%
#model/max_unpooling2d_1/range/delta?
model/max_unpooling2d_1/rangeRange,model/max_unpooling2d_1/range/start:output:0%model/max_unpooling2d_1/Rank:output:0,model/max_unpooling2d_1/range/delta:output:0*
_output_shapes
:2
model/max_unpooling2d_1/range?
"model/max_unpooling2d_1/Prod/inputPack.model/max_unpooling2d_1/strided_slice:output:0model/max_unpooling2d_1/mul:z:0!model/max_unpooling2d_1/mul_1:z:00model/max_unpooling2d_1/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2$
"model/max_unpooling2d_1/Prod/input?
model/max_unpooling2d_1/ProdProd+model/max_unpooling2d_1/Prod/input:output:0&model/max_unpooling2d_1/range:output:0*
T0*
_output_shapes
: 2
model/max_unpooling2d_1/Prod?
'model/max_unpooling2d_1/ScatterNd/shapePack%model/max_unpooling2d_1/Prod:output:0*
N*
T0*
_output_shapes
:2)
'model/max_unpooling2d_1/ScatterNd/shape?
!model/max_unpooling2d_1/ScatterNd	ScatterNd+model/max_unpooling2d_1/ExpandDims:output:0*model/max_unpooling2d_1/Reshape_1:output:00model/max_unpooling2d_1/ScatterNd/shape:output:0*
T0*
Tindices0*#
_output_shapes
:?????????2#
!model/max_unpooling2d_1/ScatterNd?
'model/max_unpooling2d_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????@   @   @   2)
'model/max_unpooling2d_1/Reshape_2/shape?
!model/max_unpooling2d_1/Reshape_2Reshape*model/max_unpooling2d_1/ScatterNd:output:00model/max_unpooling2d_1/Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????@@@2#
!model/max_unpooling2d_1/Reshape_2?
$model/conv2d_5/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02&
$model/conv2d_5/Conv2D/ReadVariableOp?
model/conv2d_5/Conv2DConv2D*model/max_unpooling2d_1/Reshape_2:output:0,model/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
2
model/conv2d_5/Conv2D?
%model/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model/conv2d_5/BiasAdd/ReadVariableOp?
model/conv2d_5/BiasAddBiasAddmodel/conv2d_5/Conv2D:output:0-model/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@2
model/conv2d_5/BiasAdd?
*model/batch_normalization_5/ReadVariableOpReadVariableOp3model_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02,
*model/batch_normalization_5/ReadVariableOp?
,model/batch_normalization_5/ReadVariableOp_1ReadVariableOp5model_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02.
,model/batch_normalization_5/ReadVariableOp_1?
;model/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02=
;model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
=model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02?
=model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
,model/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3model/conv2d_5/BiasAdd:output:02model/batch_normalization_5/ReadVariableOp:value:04model/batch_normalization_5/ReadVariableOp_1:value:0Cmodel/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
is_training( 2.
,model/batch_normalization_5/FusedBatchNormV3?
model/activation_5/ReluRelu0model/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@@2
model/activation_5/Relu?
model/max_unpooling2d_2/CastCast*model/max_pooling_with_argmax2d_1/Cast:y:0*

DstT0*

SrcT0*/
_output_shapes
:?????????@@@2
model/max_unpooling2d_2/Cast?
model/max_unpooling2d_2/ShapeShape%model/activation_5/Relu:activations:0*
T0*
_output_shapes
:2
model/max_unpooling2d_2/Shape?
+model/max_unpooling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+model/max_unpooling2d_2/strided_slice/stack?
-model/max_unpooling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-model/max_unpooling2d_2/strided_slice/stack_1?
-model/max_unpooling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-model/max_unpooling2d_2/strided_slice/stack_2?
%model/max_unpooling2d_2/strided_sliceStridedSlice&model/max_unpooling2d_2/Shape:output:04model/max_unpooling2d_2/strided_slice/stack:output:06model/max_unpooling2d_2/strided_slice/stack_1:output:06model/max_unpooling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%model/max_unpooling2d_2/strided_slice?
-model/max_unpooling2d_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-model/max_unpooling2d_2/strided_slice_1/stack?
/model/max_unpooling2d_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/model/max_unpooling2d_2/strided_slice_1/stack_1?
/model/max_unpooling2d_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/model/max_unpooling2d_2/strided_slice_1/stack_2?
'model/max_unpooling2d_2/strided_slice_1StridedSlice&model/max_unpooling2d_2/Shape:output:06model/max_unpooling2d_2/strided_slice_1/stack:output:08model/max_unpooling2d_2/strided_slice_1/stack_1:output:08model/max_unpooling2d_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'model/max_unpooling2d_2/strided_slice_1?
model/max_unpooling2d_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
model/max_unpooling2d_2/mul/y?
model/max_unpooling2d_2/mulMul0model/max_unpooling2d_2/strided_slice_1:output:0&model/max_unpooling2d_2/mul/y:output:0*
T0*
_output_shapes
: 2
model/max_unpooling2d_2/mul?
-model/max_unpooling2d_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-model/max_unpooling2d_2/strided_slice_2/stack?
/model/max_unpooling2d_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/model/max_unpooling2d_2/strided_slice_2/stack_1?
/model/max_unpooling2d_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/model/max_unpooling2d_2/strided_slice_2/stack_2?
'model/max_unpooling2d_2/strided_slice_2StridedSlice&model/max_unpooling2d_2/Shape:output:06model/max_unpooling2d_2/strided_slice_2/stack:output:08model/max_unpooling2d_2/strided_slice_2/stack_1:output:08model/max_unpooling2d_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'model/max_unpooling2d_2/strided_slice_2?
model/max_unpooling2d_2/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
model/max_unpooling2d_2/mul_1/y?
model/max_unpooling2d_2/mul_1Mul0model/max_unpooling2d_2/strided_slice_2:output:0(model/max_unpooling2d_2/mul_1/y:output:0*
T0*
_output_shapes
: 2
model/max_unpooling2d_2/mul_1?
-model/max_unpooling2d_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-model/max_unpooling2d_2/strided_slice_3/stack?
/model/max_unpooling2d_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/model/max_unpooling2d_2/strided_slice_3/stack_1?
/model/max_unpooling2d_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/model/max_unpooling2d_2/strided_slice_3/stack_2?
'model/max_unpooling2d_2/strided_slice_3StridedSlice&model/max_unpooling2d_2/Shape:output:06model/max_unpooling2d_2/strided_slice_3/stack:output:08model/max_unpooling2d_2/strided_slice_3/stack_1:output:08model/max_unpooling2d_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'model/max_unpooling2d_2/strided_slice_3?
%model/max_unpooling2d_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2'
%model/max_unpooling2d_2/Reshape/shape?
model/max_unpooling2d_2/ReshapeReshape model/max_unpooling2d_2/Cast:y:0.model/max_unpooling2d_2/Reshape/shape:output:0*
T0*#
_output_shapes
:?????????2!
model/max_unpooling2d_2/Reshape?
&model/max_unpooling2d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&model/max_unpooling2d_2/ExpandDims/dim?
"model/max_unpooling2d_2/ExpandDims
ExpandDims(model/max_unpooling2d_2/Reshape:output:0/model/max_unpooling2d_2/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2$
"model/max_unpooling2d_2/ExpandDims?
'model/max_unpooling2d_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2)
'model/max_unpooling2d_2/Reshape_1/shape?
!model/max_unpooling2d_2/Reshape_1Reshape%model/activation_5/Relu:activations:00model/max_unpooling2d_2/Reshape_1/shape:output:0*
T0*#
_output_shapes
:?????????2#
!model/max_unpooling2d_2/Reshape_1?
#model/max_unpooling2d_2/Rank/packedPack.model/max_unpooling2d_2/strided_slice:output:0model/max_unpooling2d_2/mul:z:0!model/max_unpooling2d_2/mul_1:z:00model/max_unpooling2d_2/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2%
#model/max_unpooling2d_2/Rank/packed~
model/max_unpooling2d_2/RankConst*
_output_shapes
: *
dtype0*
value	B :2
model/max_unpooling2d_2/Rank?
#model/max_unpooling2d_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2%
#model/max_unpooling2d_2/range/start?
#model/max_unpooling2d_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2%
#model/max_unpooling2d_2/range/delta?
model/max_unpooling2d_2/rangeRange,model/max_unpooling2d_2/range/start:output:0%model/max_unpooling2d_2/Rank:output:0,model/max_unpooling2d_2/range/delta:output:0*
_output_shapes
:2
model/max_unpooling2d_2/range?
"model/max_unpooling2d_2/Prod/inputPack.model/max_unpooling2d_2/strided_slice:output:0model/max_unpooling2d_2/mul:z:0!model/max_unpooling2d_2/mul_1:z:00model/max_unpooling2d_2/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2$
"model/max_unpooling2d_2/Prod/input?
model/max_unpooling2d_2/ProdProd+model/max_unpooling2d_2/Prod/input:output:0&model/max_unpooling2d_2/range:output:0*
T0*
_output_shapes
: 2
model/max_unpooling2d_2/Prod?
'model/max_unpooling2d_2/ScatterNd/shapePack%model/max_unpooling2d_2/Prod:output:0*
N*
T0*
_output_shapes
:2)
'model/max_unpooling2d_2/ScatterNd/shape?
!model/max_unpooling2d_2/ScatterNd	ScatterNd+model/max_unpooling2d_2/ExpandDims:output:0*model/max_unpooling2d_2/Reshape_1:output:00model/max_unpooling2d_2/ScatterNd/shape:output:0*
T0*
Tindices0*#
_output_shapes
:?????????2#
!model/max_unpooling2d_2/ScatterNd?
'model/max_unpooling2d_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2)
'model/max_unpooling2d_2/Reshape_2/shape?
!model/max_unpooling2d_2/Reshape_2Reshape*model/max_unpooling2d_2/ScatterNd:output:00model/max_unpooling2d_2/Reshape_2/shape:output:0*
T0*1
_output_shapes
:???????????@2#
!model/max_unpooling2d_2/Reshape_2?
$model/conv2d_6/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02&
$model/conv2d_6/Conv2D/ReadVariableOp?
model/conv2d_6/Conv2DConv2D*model/max_unpooling2d_2/Reshape_2:output:0,model/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
model/conv2d_6/Conv2D?
%model/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%model/conv2d_6/BiasAdd/ReadVariableOp?
model/conv2d_6/BiasAddBiasAddmodel/conv2d_6/Conv2D:output:0-model/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
model/conv2d_6/BiasAdd?
*model/batch_normalization_6/ReadVariableOpReadVariableOp3model_batch_normalization_6_readvariableop_resource*
_output_shapes
: *
dtype02,
*model/batch_normalization_6/ReadVariableOp?
,model/batch_normalization_6/ReadVariableOp_1ReadVariableOp5model_batch_normalization_6_readvariableop_1_resource*
_output_shapes
: *
dtype02.
,model/batch_normalization_6/ReadVariableOp_1?
;model/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02=
;model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
=model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02?
=model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
,model/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3model/conv2d_6/BiasAdd:output:02model/batch_normalization_6/ReadVariableOp:value:04model/batch_normalization_6/ReadVariableOp_1:value:0Cmodel/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2.
,model/batch_normalization_6/FusedBatchNormV3?
model/activation_6/ReluRelu0model/batch_normalization_6/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
model/activation_6/Relu?
model/max_unpooling2d_3/CastCast(model/max_pooling_with_argmax2d/Cast:y:0*

DstT0*

SrcT0*1
_output_shapes
:??????????? 2
model/max_unpooling2d_3/Cast?
model/max_unpooling2d_3/ShapeShape%model/activation_6/Relu:activations:0*
T0*
_output_shapes
:2
model/max_unpooling2d_3/Shape?
+model/max_unpooling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+model/max_unpooling2d_3/strided_slice/stack?
-model/max_unpooling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-model/max_unpooling2d_3/strided_slice/stack_1?
-model/max_unpooling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-model/max_unpooling2d_3/strided_slice/stack_2?
%model/max_unpooling2d_3/strided_sliceStridedSlice&model/max_unpooling2d_3/Shape:output:04model/max_unpooling2d_3/strided_slice/stack:output:06model/max_unpooling2d_3/strided_slice/stack_1:output:06model/max_unpooling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%model/max_unpooling2d_3/strided_slice?
-model/max_unpooling2d_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-model/max_unpooling2d_3/strided_slice_1/stack?
/model/max_unpooling2d_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/model/max_unpooling2d_3/strided_slice_1/stack_1?
/model/max_unpooling2d_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/model/max_unpooling2d_3/strided_slice_1/stack_2?
'model/max_unpooling2d_3/strided_slice_1StridedSlice&model/max_unpooling2d_3/Shape:output:06model/max_unpooling2d_3/strided_slice_1/stack:output:08model/max_unpooling2d_3/strided_slice_1/stack_1:output:08model/max_unpooling2d_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'model/max_unpooling2d_3/strided_slice_1?
model/max_unpooling2d_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
model/max_unpooling2d_3/mul/y?
model/max_unpooling2d_3/mulMul0model/max_unpooling2d_3/strided_slice_1:output:0&model/max_unpooling2d_3/mul/y:output:0*
T0*
_output_shapes
: 2
model/max_unpooling2d_3/mul?
-model/max_unpooling2d_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-model/max_unpooling2d_3/strided_slice_2/stack?
/model/max_unpooling2d_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/model/max_unpooling2d_3/strided_slice_2/stack_1?
/model/max_unpooling2d_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/model/max_unpooling2d_3/strided_slice_2/stack_2?
'model/max_unpooling2d_3/strided_slice_2StridedSlice&model/max_unpooling2d_3/Shape:output:06model/max_unpooling2d_3/strided_slice_2/stack:output:08model/max_unpooling2d_3/strided_slice_2/stack_1:output:08model/max_unpooling2d_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'model/max_unpooling2d_3/strided_slice_2?
model/max_unpooling2d_3/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
model/max_unpooling2d_3/mul_1/y?
model/max_unpooling2d_3/mul_1Mul0model/max_unpooling2d_3/strided_slice_2:output:0(model/max_unpooling2d_3/mul_1/y:output:0*
T0*
_output_shapes
: 2
model/max_unpooling2d_3/mul_1?
-model/max_unpooling2d_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-model/max_unpooling2d_3/strided_slice_3/stack?
/model/max_unpooling2d_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/model/max_unpooling2d_3/strided_slice_3/stack_1?
/model/max_unpooling2d_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/model/max_unpooling2d_3/strided_slice_3/stack_2?
'model/max_unpooling2d_3/strided_slice_3StridedSlice&model/max_unpooling2d_3/Shape:output:06model/max_unpooling2d_3/strided_slice_3/stack:output:08model/max_unpooling2d_3/strided_slice_3/stack_1:output:08model/max_unpooling2d_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'model/max_unpooling2d_3/strided_slice_3?
%model/max_unpooling2d_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2'
%model/max_unpooling2d_3/Reshape/shape?
model/max_unpooling2d_3/ReshapeReshape model/max_unpooling2d_3/Cast:y:0.model/max_unpooling2d_3/Reshape/shape:output:0*
T0*#
_output_shapes
:?????????2!
model/max_unpooling2d_3/Reshape?
&model/max_unpooling2d_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&model/max_unpooling2d_3/ExpandDims/dim?
"model/max_unpooling2d_3/ExpandDims
ExpandDims(model/max_unpooling2d_3/Reshape:output:0/model/max_unpooling2d_3/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2$
"model/max_unpooling2d_3/ExpandDims?
'model/max_unpooling2d_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2)
'model/max_unpooling2d_3/Reshape_1/shape?
!model/max_unpooling2d_3/Reshape_1Reshape%model/activation_6/Relu:activations:00model/max_unpooling2d_3/Reshape_1/shape:output:0*
T0*#
_output_shapes
:?????????2#
!model/max_unpooling2d_3/Reshape_1?
#model/max_unpooling2d_3/Rank/packedPack.model/max_unpooling2d_3/strided_slice:output:0model/max_unpooling2d_3/mul:z:0!model/max_unpooling2d_3/mul_1:z:00model/max_unpooling2d_3/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2%
#model/max_unpooling2d_3/Rank/packed~
model/max_unpooling2d_3/RankConst*
_output_shapes
: *
dtype0*
value	B :2
model/max_unpooling2d_3/Rank?
#model/max_unpooling2d_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2%
#model/max_unpooling2d_3/range/start?
#model/max_unpooling2d_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2%
#model/max_unpooling2d_3/range/delta?
model/max_unpooling2d_3/rangeRange,model/max_unpooling2d_3/range/start:output:0%model/max_unpooling2d_3/Rank:output:0,model/max_unpooling2d_3/range/delta:output:0*
_output_shapes
:2
model/max_unpooling2d_3/range?
"model/max_unpooling2d_3/Prod/inputPack.model/max_unpooling2d_3/strided_slice:output:0model/max_unpooling2d_3/mul:z:0!model/max_unpooling2d_3/mul_1:z:00model/max_unpooling2d_3/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2$
"model/max_unpooling2d_3/Prod/input?
model/max_unpooling2d_3/ProdProd+model/max_unpooling2d_3/Prod/input:output:0&model/max_unpooling2d_3/range:output:0*
T0*
_output_shapes
: 2
model/max_unpooling2d_3/Prod?
'model/max_unpooling2d_3/ScatterNd/shapePack%model/max_unpooling2d_3/Prod:output:0*
N*
T0*
_output_shapes
:2)
'model/max_unpooling2d_3/ScatterNd/shape?
!model/max_unpooling2d_3/ScatterNd	ScatterNd+model/max_unpooling2d_3/ExpandDims:output:0*model/max_unpooling2d_3/Reshape_1:output:00model/max_unpooling2d_3/ScatterNd/shape:output:0*
T0*
Tindices0*#
_output_shapes
:?????????2#
!model/max_unpooling2d_3/ScatterNd?
'model/max_unpooling2d_3/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????          2)
'model/max_unpooling2d_3/Reshape_2/shape?
!model/max_unpooling2d_3/Reshape_2Reshape*model/max_unpooling2d_3/ScatterNd:output:00model/max_unpooling2d_3/Reshape_2/shape:output:0*
T0*1
_output_shapes
:??????????? 2#
!model/max_unpooling2d_3/Reshape_2?
$model/conv2d_7/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02&
$model/conv2d_7/Conv2D/ReadVariableOp?
model/conv2d_7/Conv2DConv2D*model/max_unpooling2d_3/Reshape_2:output:0,model/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
model/conv2d_7/Conv2D?
%model/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model/conv2d_7/BiasAdd/ReadVariableOp?
model/conv2d_7/BiasAddBiasAddmodel/conv2d_7/Conv2D:output:0-model/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
model/conv2d_7/BiasAdd?
*model/batch_normalization_7/ReadVariableOpReadVariableOp3model_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype02,
*model/batch_normalization_7/ReadVariableOp?
,model/batch_normalization_7/ReadVariableOp_1ReadVariableOp5model_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,model/batch_normalization_7/ReadVariableOp_1?
;model/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02=
;model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
=model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02?
=model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
,model/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3model/conv2d_7/BiasAdd:output:02model/batch_normalization_7/ReadVariableOp:value:04model/batch_normalization_7/ReadVariableOp_1:value:0Cmodel/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2.
,model/batch_normalization_7/FusedBatchNormV3?
model/activation_7/SigmoidSigmoid0model/batch_normalization_7/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
model/activation_7/Sigmoid?
IdentityIdentitymodel/activation_7/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp:^model/batch_normalization/FusedBatchNormV3/ReadVariableOp<^model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1)^model/batch_normalization/ReadVariableOp+^model/batch_normalization/ReadVariableOp_1<^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_1/ReadVariableOp-^model/batch_normalization_1/ReadVariableOp_1<^model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_2/ReadVariableOp-^model/batch_normalization_2/ReadVariableOp_1<^model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_3/ReadVariableOp-^model/batch_normalization_3/ReadVariableOp_1<^model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_4/ReadVariableOp-^model/batch_normalization_4/ReadVariableOp_1<^model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_5/ReadVariableOp-^model/batch_normalization_5/ReadVariableOp_1<^model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_6/ReadVariableOp-^model/batch_normalization_6/ReadVariableOp_1<^model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_7/ReadVariableOp-^model/batch_normalization_7/ReadVariableOp_1$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp&^model/conv2d_3/BiasAdd/ReadVariableOp%^model/conv2d_3/Conv2D/ReadVariableOp&^model/conv2d_4/BiasAdd/ReadVariableOp%^model/conv2d_4/Conv2D/ReadVariableOp&^model/conv2d_5/BiasAdd/ReadVariableOp%^model/conv2d_5/Conv2D/ReadVariableOp&^model/conv2d_6/BiasAdd/ReadVariableOp%^model/conv2d_6/Conv2D/ReadVariableOp&^model/conv2d_7/BiasAdd/ReadVariableOp%^model/conv2d_7/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes
}:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2v
9model/batch_normalization/FusedBatchNormV3/ReadVariableOp9model/batch_normalization/FusedBatchNormV3/ReadVariableOp2z
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_12T
(model/batch_normalization/ReadVariableOp(model/batch_normalization/ReadVariableOp2X
*model/batch_normalization/ReadVariableOp_1*model/batch_normalization/ReadVariableOp_12z
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_1/ReadVariableOp*model/batch_normalization_1/ReadVariableOp2\
,model/batch_normalization_1/ReadVariableOp_1,model/batch_normalization_1/ReadVariableOp_12z
;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_2/ReadVariableOp*model/batch_normalization_2/ReadVariableOp2\
,model/batch_normalization_2/ReadVariableOp_1,model/batch_normalization_2/ReadVariableOp_12z
;model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_3/ReadVariableOp*model/batch_normalization_3/ReadVariableOp2\
,model/batch_normalization_3/ReadVariableOp_1,model/batch_normalization_3/ReadVariableOp_12z
;model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_4/ReadVariableOp*model/batch_normalization_4/ReadVariableOp2\
,model/batch_normalization_4/ReadVariableOp_1,model/batch_normalization_4/ReadVariableOp_12z
;model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_5/ReadVariableOp*model/batch_normalization_5/ReadVariableOp2\
,model/batch_normalization_5/ReadVariableOp_1,model/batch_normalization_5/ReadVariableOp_12z
;model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_6/ReadVariableOp*model/batch_normalization_6/ReadVariableOp2\
,model/batch_normalization_6/ReadVariableOp_1,model/batch_normalization_6/ReadVariableOp_12z
;model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_7/ReadVariableOp*model/batch_normalization_7/ReadVariableOp2\
,model/batch_normalization_7/ReadVariableOp_1,model/batch_normalization_7/ReadVariableOp_12J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2N
%model/conv2d_2/BiasAdd/ReadVariableOp%model/conv2d_2/BiasAdd/ReadVariableOp2L
$model/conv2d_2/Conv2D/ReadVariableOp$model/conv2d_2/Conv2D/ReadVariableOp2N
%model/conv2d_3/BiasAdd/ReadVariableOp%model/conv2d_3/BiasAdd/ReadVariableOp2L
$model/conv2d_3/Conv2D/ReadVariableOp$model/conv2d_3/Conv2D/ReadVariableOp2N
%model/conv2d_4/BiasAdd/ReadVariableOp%model/conv2d_4/BiasAdd/ReadVariableOp2L
$model/conv2d_4/Conv2D/ReadVariableOp$model/conv2d_4/Conv2D/ReadVariableOp2N
%model/conv2d_5/BiasAdd/ReadVariableOp%model/conv2d_5/BiasAdd/ReadVariableOp2L
$model/conv2d_5/Conv2D/ReadVariableOp$model/conv2d_5/Conv2D/ReadVariableOp2N
%model/conv2d_6/BiasAdd/ReadVariableOp%model/conv2d_6/BiasAdd/ReadVariableOp2L
$model/conv2d_6/Conv2D/ReadVariableOp$model/conv2d_6/Conv2D/ReadVariableOp2N
%model/conv2d_7/BiasAdd/ReadVariableOp%model/conv2d_7/BiasAdd/ReadVariableOp2L
$model/conv2d_7/Conv2D/ReadVariableOp$model/conv2d_7/Conv2D/ReadVariableOp:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_5878

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_6882

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?	
?
4__inference_batch_normalization_2_layer_call_fn_9867

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_56262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
B__inference_conv2d_7_layer_call_and_return_conditional_losses_6955

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
'__inference_conv2d_2_layer_call_fn_9782

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_64992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_7335

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  @2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
b
F__inference_activation_1_layer_call_and_return_conditional_losses_9728

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
U__inference_max_pooling_with_argmax2d_2_layer_call_and_return_conditional_losses_9932

inputs
identity

identity_1?
MaxPoolWithArgmaxMaxPoolWithArgmaxinputs*
T0*J
_output_shapes8
6:?????????  @:?????????  @*
ksize
*
paddingSAME*
strides
2
MaxPoolWithArgmaxy
CastCastMaxPoolWithArgmax:argmax:0*

DstT0*

SrcT0	*/
_output_shapes
:?????????  @2
Castv
IdentityIdentityMaxPoolWithArgmax:output:0*
T0*/
_output_shapes
:?????????  @2

Identityh

Identity_1IdentityCast:y:0*
T0*/
_output_shapes
:?????????  @2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@@:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
b
F__inference_activation_2_layer_call_and_return_conditional_losses_9911

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@@:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
c
G__inference_activation_6_layer_call_and_return_conditional_losses_10733

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:??????????? 2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
U__inference_max_pooling_with_argmax2d_1_layer_call_and_return_conditional_losses_9741

inputs
identity

identity_1?
MaxPoolWithArgmaxMaxPoolWithArgmaxinputs*
T0*J
_output_shapes8
6:?????????@@@:?????????@@@*
ksize
*
paddingSAME*
strides
2
MaxPoolWithArgmaxy
CastCastMaxPoolWithArgmax:argmax:0*

DstT0*

SrcT0	*/
_output_shapes
:?????????@@@2
Castv
IdentityIdentityMaxPoolWithArgmax:output:0*
T0*/
_output_shapes
:?????????@@@2

Identityh

Identity_1IdentityCast:y:0*
T0*/
_output_shapes
:?????????@@@2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
f
:__inference_max_pooling_with_argmax2d_1_layer_call_fn_9763

inputs
identity

identity_1?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:?????????@@@:?????????@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_max_pooling_with_argmax2d_1_layer_call_and_return_conditional_losses_75472
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@@@2

Identityx

Identity_1IdentityPartitionedCall:output:1*
T0*/
_output_shapes
:?????????@@@2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
@__inference_conv2d_layer_call_and_return_conditional_losses_9407

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?+
u
K__inference_max_unpooling2d_3_layer_call_and_return_conditional_losses_6943

inputs
inputs_1
identityi
CastCastinputs_1*

DstT0*

SrcT0*1
_output_shapes
:??????????? 2
CastD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2T
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapem
ReshapeReshapeCast:y:0Reshape/shape:output:0*
T0*#
_output_shapes
:?????????2	
Reshapek
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim?

ExpandDims
ExpandDimsReshape:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2

ExpandDimsu
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_1/shapeq
	Reshape_1ReshapeinputsReshape_1/shape:output:0*
T0*#
_output_shapes
:?????????2
	Reshape_1?
Rank/packedPackstrided_slice:output:0mul:z:0	mul_1:z:0strided_slice_3:output:0*
N*
T0*
_output_shapes
:2
Rank/packedN
RankConst*
_output_shapes
: *
dtype0*
value	B :2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltan
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:2
range?

Prod/inputPackstrided_slice:output:0mul:z:0	mul_1:z:0strided_slice_3:output:0*
N*
T0*
_output_shapes
:2

Prod/inputZ
ProdProdProd/input:output:0range:output:0*
T0*
_output_shapes
: 2
Prodg
ScatterNd/shapePackProd:output:0*
N*
T0*
_output_shapes
:2
ScatterNd/shape?
	ScatterNd	ScatterNdExpandDims:output:0Reshape_1:output:0ScatterNd/shape:output:0*
T0*
Tindices0*#
_output_shapes
:?????????2
	ScatterNd{
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????          2
Reshape_2/shape?
	Reshape_2ReshapeScatterNd:output:0Reshape_2/shape:output:0*
T0*1
_output_shapes
:??????????? 2
	Reshape_2p
IdentityIdentityReshape_2:output:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::??????????? :??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs:YU
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_10216

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
5__inference_batch_normalization_6_layer_call_fn_10715

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_68822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_10252

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  @2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_10419

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
c
G__inference_activation_7_layer_call_and_return_conditional_losses_10936

inputs
identitya
SigmoidSigmoidinputs*
T0*1
_output_shapes
:???????????2	
Sigmoidi
IdentityIdentitySigmoid:y:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?/
?__inference_model_layer_call_and_return_conditional_losses_9195

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 9
+batch_normalization_readvariableop_resource: ;
-batch_normalization_readvariableop_1_resource: J
<batch_normalization_fusedbatchnormv3_readvariableop_resource: L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_1_conv2d_readvariableop_resource: @6
(conv2d_1_biasadd_readvariableop_resource:@;
-batch_normalization_1_readvariableop_resource:@=
/batch_normalization_1_readvariableop_1_resource:@L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@A
'conv2d_2_conv2d_readvariableop_resource:@@6
(conv2d_2_biasadd_readvariableop_resource:@;
-batch_normalization_2_readvariableop_resource:@=
/batch_normalization_2_readvariableop_1_resource:@L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@B
'conv2d_3_conv2d_readvariableop_resource:@?7
(conv2d_3_biasadd_readvariableop_resource:	?<
-batch_normalization_3_readvariableop_resource:	?>
/batch_normalization_3_readvariableop_1_resource:	?M
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	?O
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	?B
'conv2d_4_conv2d_readvariableop_resource:?@6
(conv2d_4_biasadd_readvariableop_resource:@;
-batch_normalization_4_readvariableop_resource:@=
/batch_normalization_4_readvariableop_1_resource:@L
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:@A
'conv2d_5_conv2d_readvariableop_resource:@@6
(conv2d_5_biasadd_readvariableop_resource:@;
-batch_normalization_5_readvariableop_resource:@=
/batch_normalization_5_readvariableop_1_resource:@L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@A
'conv2d_6_conv2d_readvariableop_resource:@ 6
(conv2d_6_biasadd_readvariableop_resource: ;
-batch_normalization_6_readvariableop_resource: =
/batch_normalization_6_readvariableop_1_resource: L
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_7_conv2d_readvariableop_resource: 6
(conv2d_7_biasadd_readvariableop_resource:;
-batch_normalization_7_readvariableop_resource:=
/batch_normalization_7_readvariableop_1_resource:L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:
identity??"batch_normalization/AssignNewValue?$batch_normalization/AssignNewValue_1?3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?$batch_normalization_1/AssignNewValue?&batch_normalization_1/AssignNewValue_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?$batch_normalization_2/AssignNewValue?&batch_normalization_2/AssignNewValue_1?5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_2/ReadVariableOp?&batch_normalization_2/ReadVariableOp_1?$batch_normalization_3/AssignNewValue?&batch_normalization_3/AssignNewValue_1?5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_3/ReadVariableOp?&batch_normalization_3/ReadVariableOp_1?$batch_normalization_4/AssignNewValue?&batch_normalization_4/AssignNewValue_1?5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_4/ReadVariableOp?&batch_normalization_4/ReadVariableOp_1?$batch_normalization_5/AssignNewValue?&batch_normalization_5/AssignNewValue_1?5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_5/ReadVariableOp?&batch_normalization_5/ReadVariableOp_1?$batch_normalization_6/AssignNewValue?&batch_normalization_6/AssignNewValue_1?5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_6/ReadVariableOp?&batch_normalization_6/ReadVariableOp_1?$batch_normalization_7/AssignNewValue?&batch_normalization_7/AssignNewValue_1?5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_7/ReadVariableOp?&batch_normalization_7/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2&
$batch_normalization/FusedBatchNormV3?
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue?
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1?
activation/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
activation/Relu?
+max_pooling_with_argmax2d/MaxPoolWithArgmaxMaxPoolWithArgmaxactivation/Relu:activations:0*
T0*N
_output_shapes<
::??????????? :??????????? *
ksize
*
paddingSAME*
strides
2-
+max_pooling_with_argmax2d/MaxPoolWithArgmax?
max_pooling_with_argmax2d/CastCast4max_pooling_with_argmax2d/MaxPoolWithArgmax:argmax:0*

DstT0*

SrcT0	*1
_output_shapes
:??????????? 2 
max_pooling_with_argmax2d/Cast?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2D4max_pooling_with_argmax2d/MaxPoolWithArgmax:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
conv2d_1/BiasAdd?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_1/FusedBatchNormV3?
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue?
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1?
activation_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????@2
activation_1/Relu?
-max_pooling_with_argmax2d_1/MaxPoolWithArgmaxMaxPoolWithArgmaxactivation_1/Relu:activations:0*
T0*J
_output_shapes8
6:?????????@@@:?????????@@@*
ksize
*
paddingSAME*
strides
2/
-max_pooling_with_argmax2d_1/MaxPoolWithArgmax?
 max_pooling_with_argmax2d_1/CastCast6max_pooling_with_argmax2d_1/MaxPoolWithArgmax:argmax:0*

DstT0*

SrcT0	*/
_output_shapes
:?????????@@@2"
 max_pooling_with_argmax2d_1/Cast?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D6max_pooling_with_argmax2d_1/MaxPoolWithArgmax:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@2
conv2d_2/BiasAdd?
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp?
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1?
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_2/FusedBatchNormV3?
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValue?
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1?
activation_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@@2
activation_2/Relu?
-max_pooling_with_argmax2d_2/MaxPoolWithArgmaxMaxPoolWithArgmaxactivation_2/Relu:activations:0*
T0*J
_output_shapes8
6:?????????  @:?????????  @*
ksize
*
paddingSAME*
strides
2/
-max_pooling_with_argmax2d_2/MaxPoolWithArgmax?
 max_pooling_with_argmax2d_2/CastCast6max_pooling_with_argmax2d_2/MaxPoolWithArgmax:argmax:0*

DstT0*

SrcT0	*/
_output_shapes
:?????????  @2"
 max_pooling_with_argmax2d_2/Cast?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2D6max_pooling_with_argmax2d_2/MaxPoolWithArgmax:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?2
conv2d_3/BiasAdd?
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_3/ReadVariableOp?
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_3/ReadVariableOp_1?
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_3/FusedBatchNormV3?
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValue?
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1?
activation_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????  ?2
activation_3/Relu?
-max_pooling_with_argmax2d_3/MaxPoolWithArgmaxMaxPoolWithArgmaxactivation_3/Relu:activations:0*
T0*L
_output_shapes:
8:??????????:??????????*
ksize
*
paddingSAME*
strides
2/
-max_pooling_with_argmax2d_3/MaxPoolWithArgmax?
 max_pooling_with_argmax2d_3/CastCast6max_pooling_with_argmax2d_3/MaxPoolWithArgmax:argmax:0*

DstT0*

SrcT0	*0
_output_shapes
:??????????2"
 max_pooling_with_argmax2d_3/Cast?
max_unpooling2d/CastCast$max_pooling_with_argmax2d_3/Cast:y:0*

DstT0*

SrcT0*0
_output_shapes
:??????????2
max_unpooling2d/Cast?
max_unpooling2d/ShapeShape6max_pooling_with_argmax2d_3/MaxPoolWithArgmax:output:0*
T0*
_output_shapes
:2
max_unpooling2d/Shape?
#max_unpooling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#max_unpooling2d/strided_slice/stack?
%max_unpooling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%max_unpooling2d/strided_slice/stack_1?
%max_unpooling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%max_unpooling2d/strided_slice/stack_2?
max_unpooling2d/strided_sliceStridedSlicemax_unpooling2d/Shape:output:0,max_unpooling2d/strided_slice/stack:output:0.max_unpooling2d/strided_slice/stack_1:output:0.max_unpooling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
max_unpooling2d/strided_slice?
%max_unpooling2d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2'
%max_unpooling2d/strided_slice_1/stack?
'max_unpooling2d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d/strided_slice_1/stack_1?
'max_unpooling2d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d/strided_slice_1/stack_2?
max_unpooling2d/strided_slice_1StridedSlicemax_unpooling2d/Shape:output:0.max_unpooling2d/strided_slice_1/stack:output:00max_unpooling2d/strided_slice_1/stack_1:output:00max_unpooling2d/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
max_unpooling2d/strided_slice_1p
max_unpooling2d/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d/mul/y?
max_unpooling2d/mulMul(max_unpooling2d/strided_slice_1:output:0max_unpooling2d/mul/y:output:0*
T0*
_output_shapes
: 2
max_unpooling2d/mul?
%max_unpooling2d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2'
%max_unpooling2d/strided_slice_2/stack?
'max_unpooling2d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d/strided_slice_2/stack_1?
'max_unpooling2d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d/strided_slice_2/stack_2?
max_unpooling2d/strided_slice_2StridedSlicemax_unpooling2d/Shape:output:0.max_unpooling2d/strided_slice_2/stack:output:00max_unpooling2d/strided_slice_2/stack_1:output:00max_unpooling2d/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
max_unpooling2d/strided_slice_2t
max_unpooling2d/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d/mul_1/y?
max_unpooling2d/mul_1Mul(max_unpooling2d/strided_slice_2:output:0 max_unpooling2d/mul_1/y:output:0*
T0*
_output_shapes
: 2
max_unpooling2d/mul_1?
%max_unpooling2d/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2'
%max_unpooling2d/strided_slice_3/stack?
'max_unpooling2d/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d/strided_slice_3/stack_1?
'max_unpooling2d/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d/strided_slice_3/stack_2?
max_unpooling2d/strided_slice_3StridedSlicemax_unpooling2d/Shape:output:0.max_unpooling2d/strided_slice_3/stack:output:00max_unpooling2d/strided_slice_3/stack_1:output:00max_unpooling2d/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
max_unpooling2d/strided_slice_3?
max_unpooling2d/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
max_unpooling2d/Reshape/shape?
max_unpooling2d/ReshapeReshapemax_unpooling2d/Cast:y:0&max_unpooling2d/Reshape/shape:output:0*
T0*#
_output_shapes
:?????????2
max_unpooling2d/Reshape?
max_unpooling2d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
max_unpooling2d/ExpandDims/dim?
max_unpooling2d/ExpandDims
ExpandDims max_unpooling2d/Reshape:output:0'max_unpooling2d/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2
max_unpooling2d/ExpandDims?
max_unpooling2d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2!
max_unpooling2d/Reshape_1/shape?
max_unpooling2d/Reshape_1Reshape6max_pooling_with_argmax2d_3/MaxPoolWithArgmax:output:0(max_unpooling2d/Reshape_1/shape:output:0*
T0*#
_output_shapes
:?????????2
max_unpooling2d/Reshape_1?
max_unpooling2d/Rank/packedPack&max_unpooling2d/strided_slice:output:0max_unpooling2d/mul:z:0max_unpooling2d/mul_1:z:0(max_unpooling2d/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2
max_unpooling2d/Rank/packedn
max_unpooling2d/RankConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d/Rank|
max_unpooling2d/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
max_unpooling2d/range/start|
max_unpooling2d/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d/range/delta?
max_unpooling2d/rangeRange$max_unpooling2d/range/start:output:0max_unpooling2d/Rank:output:0$max_unpooling2d/range/delta:output:0*
_output_shapes
:2
max_unpooling2d/range?
max_unpooling2d/Prod/inputPack&max_unpooling2d/strided_slice:output:0max_unpooling2d/mul:z:0max_unpooling2d/mul_1:z:0(max_unpooling2d/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2
max_unpooling2d/Prod/input?
max_unpooling2d/ProdProd#max_unpooling2d/Prod/input:output:0max_unpooling2d/range:output:0*
T0*
_output_shapes
: 2
max_unpooling2d/Prod?
max_unpooling2d/ScatterNd/shapePackmax_unpooling2d/Prod:output:0*
N*
T0*
_output_shapes
:2!
max_unpooling2d/ScatterNd/shape?
max_unpooling2d/ScatterNd	ScatterNd#max_unpooling2d/ExpandDims:output:0"max_unpooling2d/Reshape_1:output:0(max_unpooling2d/ScatterNd/shape:output:0*
T0*
Tindices0*#
_output_shapes
:?????????2
max_unpooling2d/ScatterNd?
max_unpooling2d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????        ?   2!
max_unpooling2d/Reshape_2/shape?
max_unpooling2d/Reshape_2Reshape"max_unpooling2d/ScatterNd:output:0(max_unpooling2d/Reshape_2/shape:output:0*
T0*0
_output_shapes
:?????????  ?2
max_unpooling2d/Reshape_2?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2D"max_unpooling2d/Reshape_2:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2
conv2d_4/BiasAdd?
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_4/ReadVariableOp?
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_4/ReadVariableOp_1?
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_4/FusedBatchNormV3?
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValue?
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1?
activation_4/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  @2
activation_4/Relu?
max_unpooling2d_1/CastCast$max_pooling_with_argmax2d_2/Cast:y:0*

DstT0*

SrcT0*/
_output_shapes
:?????????  @2
max_unpooling2d_1/Cast?
max_unpooling2d_1/ShapeShapeactivation_4/Relu:activations:0*
T0*
_output_shapes
:2
max_unpooling2d_1/Shape?
%max_unpooling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%max_unpooling2d_1/strided_slice/stack?
'max_unpooling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d_1/strided_slice/stack_1?
'max_unpooling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d_1/strided_slice/stack_2?
max_unpooling2d_1/strided_sliceStridedSlice max_unpooling2d_1/Shape:output:0.max_unpooling2d_1/strided_slice/stack:output:00max_unpooling2d_1/strided_slice/stack_1:output:00max_unpooling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
max_unpooling2d_1/strided_slice?
'max_unpooling2d_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d_1/strided_slice_1/stack?
)max_unpooling2d_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_1/strided_slice_1/stack_1?
)max_unpooling2d_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_1/strided_slice_1/stack_2?
!max_unpooling2d_1/strided_slice_1StridedSlice max_unpooling2d_1/Shape:output:00max_unpooling2d_1/strided_slice_1/stack:output:02max_unpooling2d_1/strided_slice_1/stack_1:output:02max_unpooling2d_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!max_unpooling2d_1/strided_slice_1t
max_unpooling2d_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d_1/mul/y?
max_unpooling2d_1/mulMul*max_unpooling2d_1/strided_slice_1:output:0 max_unpooling2d_1/mul/y:output:0*
T0*
_output_shapes
: 2
max_unpooling2d_1/mul?
'max_unpooling2d_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d_1/strided_slice_2/stack?
)max_unpooling2d_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_1/strided_slice_2/stack_1?
)max_unpooling2d_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_1/strided_slice_2/stack_2?
!max_unpooling2d_1/strided_slice_2StridedSlice max_unpooling2d_1/Shape:output:00max_unpooling2d_1/strided_slice_2/stack:output:02max_unpooling2d_1/strided_slice_2/stack_1:output:02max_unpooling2d_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!max_unpooling2d_1/strided_slice_2x
max_unpooling2d_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d_1/mul_1/y?
max_unpooling2d_1/mul_1Mul*max_unpooling2d_1/strided_slice_2:output:0"max_unpooling2d_1/mul_1/y:output:0*
T0*
_output_shapes
: 2
max_unpooling2d_1/mul_1?
'max_unpooling2d_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d_1/strided_slice_3/stack?
)max_unpooling2d_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_1/strided_slice_3/stack_1?
)max_unpooling2d_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_1/strided_slice_3/stack_2?
!max_unpooling2d_1/strided_slice_3StridedSlice max_unpooling2d_1/Shape:output:00max_unpooling2d_1/strided_slice_3/stack:output:02max_unpooling2d_1/strided_slice_3/stack_1:output:02max_unpooling2d_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!max_unpooling2d_1/strided_slice_3?
max_unpooling2d_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2!
max_unpooling2d_1/Reshape/shape?
max_unpooling2d_1/ReshapeReshapemax_unpooling2d_1/Cast:y:0(max_unpooling2d_1/Reshape/shape:output:0*
T0*#
_output_shapes
:?????????2
max_unpooling2d_1/Reshape?
 max_unpooling2d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 max_unpooling2d_1/ExpandDims/dim?
max_unpooling2d_1/ExpandDims
ExpandDims"max_unpooling2d_1/Reshape:output:0)max_unpooling2d_1/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2
max_unpooling2d_1/ExpandDims?
!max_unpooling2d_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2#
!max_unpooling2d_1/Reshape_1/shape?
max_unpooling2d_1/Reshape_1Reshapeactivation_4/Relu:activations:0*max_unpooling2d_1/Reshape_1/shape:output:0*
T0*#
_output_shapes
:?????????2
max_unpooling2d_1/Reshape_1?
max_unpooling2d_1/Rank/packedPack(max_unpooling2d_1/strided_slice:output:0max_unpooling2d_1/mul:z:0max_unpooling2d_1/mul_1:z:0*max_unpooling2d_1/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2
max_unpooling2d_1/Rank/packedr
max_unpooling2d_1/RankConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d_1/Rank?
max_unpooling2d_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
max_unpooling2d_1/range/start?
max_unpooling2d_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d_1/range/delta?
max_unpooling2d_1/rangeRange&max_unpooling2d_1/range/start:output:0max_unpooling2d_1/Rank:output:0&max_unpooling2d_1/range/delta:output:0*
_output_shapes
:2
max_unpooling2d_1/range?
max_unpooling2d_1/Prod/inputPack(max_unpooling2d_1/strided_slice:output:0max_unpooling2d_1/mul:z:0max_unpooling2d_1/mul_1:z:0*max_unpooling2d_1/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2
max_unpooling2d_1/Prod/input?
max_unpooling2d_1/ProdProd%max_unpooling2d_1/Prod/input:output:0 max_unpooling2d_1/range:output:0*
T0*
_output_shapes
: 2
max_unpooling2d_1/Prod?
!max_unpooling2d_1/ScatterNd/shapePackmax_unpooling2d_1/Prod:output:0*
N*
T0*
_output_shapes
:2#
!max_unpooling2d_1/ScatterNd/shape?
max_unpooling2d_1/ScatterNd	ScatterNd%max_unpooling2d_1/ExpandDims:output:0$max_unpooling2d_1/Reshape_1:output:0*max_unpooling2d_1/ScatterNd/shape:output:0*
T0*
Tindices0*#
_output_shapes
:?????????2
max_unpooling2d_1/ScatterNd?
!max_unpooling2d_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????@   @   @   2#
!max_unpooling2d_1/Reshape_2/shape?
max_unpooling2d_1/Reshape_2Reshape$max_unpooling2d_1/ScatterNd:output:0*max_unpooling2d_1/Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????@@@2
max_unpooling2d_1/Reshape_2?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2D$max_unpooling2d_1/Reshape_2:output:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@2
conv2d_5/BiasAdd?
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_5/ReadVariableOp?
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_5/ReadVariableOp_1?
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_5/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_5/FusedBatchNormV3?
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_5/AssignNewValue?
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_5/AssignNewValue_1?
activation_5/ReluRelu*batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@@2
activation_5/Relu?
max_unpooling2d_2/CastCast$max_pooling_with_argmax2d_1/Cast:y:0*

DstT0*

SrcT0*/
_output_shapes
:?????????@@@2
max_unpooling2d_2/Cast?
max_unpooling2d_2/ShapeShapeactivation_5/Relu:activations:0*
T0*
_output_shapes
:2
max_unpooling2d_2/Shape?
%max_unpooling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%max_unpooling2d_2/strided_slice/stack?
'max_unpooling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d_2/strided_slice/stack_1?
'max_unpooling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d_2/strided_slice/stack_2?
max_unpooling2d_2/strided_sliceStridedSlice max_unpooling2d_2/Shape:output:0.max_unpooling2d_2/strided_slice/stack:output:00max_unpooling2d_2/strided_slice/stack_1:output:00max_unpooling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
max_unpooling2d_2/strided_slice?
'max_unpooling2d_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d_2/strided_slice_1/stack?
)max_unpooling2d_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_2/strided_slice_1/stack_1?
)max_unpooling2d_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_2/strided_slice_1/stack_2?
!max_unpooling2d_2/strided_slice_1StridedSlice max_unpooling2d_2/Shape:output:00max_unpooling2d_2/strided_slice_1/stack:output:02max_unpooling2d_2/strided_slice_1/stack_1:output:02max_unpooling2d_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!max_unpooling2d_2/strided_slice_1t
max_unpooling2d_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d_2/mul/y?
max_unpooling2d_2/mulMul*max_unpooling2d_2/strided_slice_1:output:0 max_unpooling2d_2/mul/y:output:0*
T0*
_output_shapes
: 2
max_unpooling2d_2/mul?
'max_unpooling2d_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d_2/strided_slice_2/stack?
)max_unpooling2d_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_2/strided_slice_2/stack_1?
)max_unpooling2d_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_2/strided_slice_2/stack_2?
!max_unpooling2d_2/strided_slice_2StridedSlice max_unpooling2d_2/Shape:output:00max_unpooling2d_2/strided_slice_2/stack:output:02max_unpooling2d_2/strided_slice_2/stack_1:output:02max_unpooling2d_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!max_unpooling2d_2/strided_slice_2x
max_unpooling2d_2/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d_2/mul_1/y?
max_unpooling2d_2/mul_1Mul*max_unpooling2d_2/strided_slice_2:output:0"max_unpooling2d_2/mul_1/y:output:0*
T0*
_output_shapes
: 2
max_unpooling2d_2/mul_1?
'max_unpooling2d_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d_2/strided_slice_3/stack?
)max_unpooling2d_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_2/strided_slice_3/stack_1?
)max_unpooling2d_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_2/strided_slice_3/stack_2?
!max_unpooling2d_2/strided_slice_3StridedSlice max_unpooling2d_2/Shape:output:00max_unpooling2d_2/strided_slice_3/stack:output:02max_unpooling2d_2/strided_slice_3/stack_1:output:02max_unpooling2d_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!max_unpooling2d_2/strided_slice_3?
max_unpooling2d_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2!
max_unpooling2d_2/Reshape/shape?
max_unpooling2d_2/ReshapeReshapemax_unpooling2d_2/Cast:y:0(max_unpooling2d_2/Reshape/shape:output:0*
T0*#
_output_shapes
:?????????2
max_unpooling2d_2/Reshape?
 max_unpooling2d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 max_unpooling2d_2/ExpandDims/dim?
max_unpooling2d_2/ExpandDims
ExpandDims"max_unpooling2d_2/Reshape:output:0)max_unpooling2d_2/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2
max_unpooling2d_2/ExpandDims?
!max_unpooling2d_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2#
!max_unpooling2d_2/Reshape_1/shape?
max_unpooling2d_2/Reshape_1Reshapeactivation_5/Relu:activations:0*max_unpooling2d_2/Reshape_1/shape:output:0*
T0*#
_output_shapes
:?????????2
max_unpooling2d_2/Reshape_1?
max_unpooling2d_2/Rank/packedPack(max_unpooling2d_2/strided_slice:output:0max_unpooling2d_2/mul:z:0max_unpooling2d_2/mul_1:z:0*max_unpooling2d_2/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2
max_unpooling2d_2/Rank/packedr
max_unpooling2d_2/RankConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d_2/Rank?
max_unpooling2d_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
max_unpooling2d_2/range/start?
max_unpooling2d_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d_2/range/delta?
max_unpooling2d_2/rangeRange&max_unpooling2d_2/range/start:output:0max_unpooling2d_2/Rank:output:0&max_unpooling2d_2/range/delta:output:0*
_output_shapes
:2
max_unpooling2d_2/range?
max_unpooling2d_2/Prod/inputPack(max_unpooling2d_2/strided_slice:output:0max_unpooling2d_2/mul:z:0max_unpooling2d_2/mul_1:z:0*max_unpooling2d_2/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2
max_unpooling2d_2/Prod/input?
max_unpooling2d_2/ProdProd%max_unpooling2d_2/Prod/input:output:0 max_unpooling2d_2/range:output:0*
T0*
_output_shapes
: 2
max_unpooling2d_2/Prod?
!max_unpooling2d_2/ScatterNd/shapePackmax_unpooling2d_2/Prod:output:0*
N*
T0*
_output_shapes
:2#
!max_unpooling2d_2/ScatterNd/shape?
max_unpooling2d_2/ScatterNd	ScatterNd%max_unpooling2d_2/ExpandDims:output:0$max_unpooling2d_2/Reshape_1:output:0*max_unpooling2d_2/ScatterNd/shape:output:0*
T0*
Tindices0*#
_output_shapes
:?????????2
max_unpooling2d_2/ScatterNd?
!max_unpooling2d_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2#
!max_unpooling2d_2/Reshape_2/shape?
max_unpooling2d_2/Reshape_2Reshape$max_unpooling2d_2/ScatterNd:output:0*max_unpooling2d_2/Reshape_2/shape:output:0*
T0*1
_output_shapes
:???????????@2
max_unpooling2d_2/Reshape_2?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2D$max_unpooling2d_2/Reshape_2:output:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_6/BiasAdd?
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_6/ReadVariableOp?
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_6/ReadVariableOp_1?
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_6/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_6/FusedBatchNormV3?
$batch_normalization_6/AssignNewValueAssignVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_6/AssignNewValue?
&batch_normalization_6/AssignNewValue_1AssignVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_6/AssignNewValue_1?
activation_6/ReluRelu*batch_normalization_6/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? 2
activation_6/Relu?
max_unpooling2d_3/CastCast"max_pooling_with_argmax2d/Cast:y:0*

DstT0*

SrcT0*1
_output_shapes
:??????????? 2
max_unpooling2d_3/Cast?
max_unpooling2d_3/ShapeShapeactivation_6/Relu:activations:0*
T0*
_output_shapes
:2
max_unpooling2d_3/Shape?
%max_unpooling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%max_unpooling2d_3/strided_slice/stack?
'max_unpooling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d_3/strided_slice/stack_1?
'max_unpooling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d_3/strided_slice/stack_2?
max_unpooling2d_3/strided_sliceStridedSlice max_unpooling2d_3/Shape:output:0.max_unpooling2d_3/strided_slice/stack:output:00max_unpooling2d_3/strided_slice/stack_1:output:00max_unpooling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
max_unpooling2d_3/strided_slice?
'max_unpooling2d_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d_3/strided_slice_1/stack?
)max_unpooling2d_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_3/strided_slice_1/stack_1?
)max_unpooling2d_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_3/strided_slice_1/stack_2?
!max_unpooling2d_3/strided_slice_1StridedSlice max_unpooling2d_3/Shape:output:00max_unpooling2d_3/strided_slice_1/stack:output:02max_unpooling2d_3/strided_slice_1/stack_1:output:02max_unpooling2d_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!max_unpooling2d_3/strided_slice_1t
max_unpooling2d_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d_3/mul/y?
max_unpooling2d_3/mulMul*max_unpooling2d_3/strided_slice_1:output:0 max_unpooling2d_3/mul/y:output:0*
T0*
_output_shapes
: 2
max_unpooling2d_3/mul?
'max_unpooling2d_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d_3/strided_slice_2/stack?
)max_unpooling2d_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_3/strided_slice_2/stack_1?
)max_unpooling2d_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_3/strided_slice_2/stack_2?
!max_unpooling2d_3/strided_slice_2StridedSlice max_unpooling2d_3/Shape:output:00max_unpooling2d_3/strided_slice_2/stack:output:02max_unpooling2d_3/strided_slice_2/stack_1:output:02max_unpooling2d_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!max_unpooling2d_3/strided_slice_2x
max_unpooling2d_3/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d_3/mul_1/y?
max_unpooling2d_3/mul_1Mul*max_unpooling2d_3/strided_slice_2:output:0"max_unpooling2d_3/mul_1/y:output:0*
T0*
_output_shapes
: 2
max_unpooling2d_3/mul_1?
'max_unpooling2d_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'max_unpooling2d_3/strided_slice_3/stack?
)max_unpooling2d_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_3/strided_slice_3/stack_1?
)max_unpooling2d_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)max_unpooling2d_3/strided_slice_3/stack_2?
!max_unpooling2d_3/strided_slice_3StridedSlice max_unpooling2d_3/Shape:output:00max_unpooling2d_3/strided_slice_3/stack:output:02max_unpooling2d_3/strided_slice_3/stack_1:output:02max_unpooling2d_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!max_unpooling2d_3/strided_slice_3?
max_unpooling2d_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2!
max_unpooling2d_3/Reshape/shape?
max_unpooling2d_3/ReshapeReshapemax_unpooling2d_3/Cast:y:0(max_unpooling2d_3/Reshape/shape:output:0*
T0*#
_output_shapes
:?????????2
max_unpooling2d_3/Reshape?
 max_unpooling2d_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 max_unpooling2d_3/ExpandDims/dim?
max_unpooling2d_3/ExpandDims
ExpandDims"max_unpooling2d_3/Reshape:output:0)max_unpooling2d_3/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2
max_unpooling2d_3/ExpandDims?
!max_unpooling2d_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2#
!max_unpooling2d_3/Reshape_1/shape?
max_unpooling2d_3/Reshape_1Reshapeactivation_6/Relu:activations:0*max_unpooling2d_3/Reshape_1/shape:output:0*
T0*#
_output_shapes
:?????????2
max_unpooling2d_3/Reshape_1?
max_unpooling2d_3/Rank/packedPack(max_unpooling2d_3/strided_slice:output:0max_unpooling2d_3/mul:z:0max_unpooling2d_3/mul_1:z:0*max_unpooling2d_3/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2
max_unpooling2d_3/Rank/packedr
max_unpooling2d_3/RankConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d_3/Rank?
max_unpooling2d_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
max_unpooling2d_3/range/start?
max_unpooling2d_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
max_unpooling2d_3/range/delta?
max_unpooling2d_3/rangeRange&max_unpooling2d_3/range/start:output:0max_unpooling2d_3/Rank:output:0&max_unpooling2d_3/range/delta:output:0*
_output_shapes
:2
max_unpooling2d_3/range?
max_unpooling2d_3/Prod/inputPack(max_unpooling2d_3/strided_slice:output:0max_unpooling2d_3/mul:z:0max_unpooling2d_3/mul_1:z:0*max_unpooling2d_3/strided_slice_3:output:0*
N*
T0*
_output_shapes
:2
max_unpooling2d_3/Prod/input?
max_unpooling2d_3/ProdProd%max_unpooling2d_3/Prod/input:output:0 max_unpooling2d_3/range:output:0*
T0*
_output_shapes
: 2
max_unpooling2d_3/Prod?
!max_unpooling2d_3/ScatterNd/shapePackmax_unpooling2d_3/Prod:output:0*
N*
T0*
_output_shapes
:2#
!max_unpooling2d_3/ScatterNd/shape?
max_unpooling2d_3/ScatterNd	ScatterNd%max_unpooling2d_3/ExpandDims:output:0$max_unpooling2d_3/Reshape_1:output:0*max_unpooling2d_3/ScatterNd/shape:output:0*
T0*
Tindices0*#
_output_shapes
:?????????2
max_unpooling2d_3/ScatterNd?
!max_unpooling2d_3/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????          2#
!max_unpooling2d_3/Reshape_2/shape?
max_unpooling2d_3/Reshape_2Reshape$max_unpooling2d_3/ScatterNd:output:0*max_unpooling2d_3/Reshape_2/shape:output:0*
T0*1
_output_shapes
:??????????? 2
max_unpooling2d_3/Reshape_2?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_7/Conv2D/ReadVariableOp?
conv2d_7/Conv2DConv2D$max_unpooling2d_3/Reshape_2:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_7/Conv2D?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_7/BiasAdd?
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_7/ReadVariableOp?
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_7/ReadVariableOp_1?
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_7/BiasAdd:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_7/FusedBatchNormV3?
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_7/AssignNewValue?
&batch_normalization_7/AssignNewValue_1AssignVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_7/AssignNewValue_1?
activation_7/SigmoidSigmoid*batch_normalization_7/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
activation_7/Sigmoid}
IdentityIdentityactivation_7/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes
}:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12L
$batch_normalization_6/AssignNewValue$batch_normalization_6/AssignNewValue2P
&batch_normalization_6/AssignNewValue_1&batch_normalization_6/AssignNewValue_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6522

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
??
?Q
!__inference__traced_restore_11700
file_prefix8
assignvariableop_conv2d_kernel: ,
assignvariableop_1_conv2d_bias: :
,assignvariableop_2_batch_normalization_gamma: 9
+assignvariableop_3_batch_normalization_beta: @
2assignvariableop_4_batch_normalization_moving_mean: D
6assignvariableop_5_batch_normalization_moving_variance: <
"assignvariableop_6_conv2d_1_kernel: @.
 assignvariableop_7_conv2d_1_bias:@<
.assignvariableop_8_batch_normalization_1_gamma:@;
-assignvariableop_9_batch_normalization_1_beta:@C
5assignvariableop_10_batch_normalization_1_moving_mean:@G
9assignvariableop_11_batch_normalization_1_moving_variance:@=
#assignvariableop_12_conv2d_2_kernel:@@/
!assignvariableop_13_conv2d_2_bias:@=
/assignvariableop_14_batch_normalization_2_gamma:@<
.assignvariableop_15_batch_normalization_2_beta:@C
5assignvariableop_16_batch_normalization_2_moving_mean:@G
9assignvariableop_17_batch_normalization_2_moving_variance:@>
#assignvariableop_18_conv2d_3_kernel:@?0
!assignvariableop_19_conv2d_3_bias:	?>
/assignvariableop_20_batch_normalization_3_gamma:	?=
.assignvariableop_21_batch_normalization_3_beta:	?D
5assignvariableop_22_batch_normalization_3_moving_mean:	?H
9assignvariableop_23_batch_normalization_3_moving_variance:	?>
#assignvariableop_24_conv2d_4_kernel:?@/
!assignvariableop_25_conv2d_4_bias:@=
/assignvariableop_26_batch_normalization_4_gamma:@<
.assignvariableop_27_batch_normalization_4_beta:@C
5assignvariableop_28_batch_normalization_4_moving_mean:@G
9assignvariableop_29_batch_normalization_4_moving_variance:@=
#assignvariableop_30_conv2d_5_kernel:@@/
!assignvariableop_31_conv2d_5_bias:@=
/assignvariableop_32_batch_normalization_5_gamma:@<
.assignvariableop_33_batch_normalization_5_beta:@C
5assignvariableop_34_batch_normalization_5_moving_mean:@G
9assignvariableop_35_batch_normalization_5_moving_variance:@=
#assignvariableop_36_conv2d_6_kernel:@ /
!assignvariableop_37_conv2d_6_bias: =
/assignvariableop_38_batch_normalization_6_gamma: <
.assignvariableop_39_batch_normalization_6_beta: C
5assignvariableop_40_batch_normalization_6_moving_mean: G
9assignvariableop_41_batch_normalization_6_moving_variance: =
#assignvariableop_42_conv2d_7_kernel: /
!assignvariableop_43_conv2d_7_bias:=
/assignvariableop_44_batch_normalization_7_gamma:<
.assignvariableop_45_batch_normalization_7_beta:C
5assignvariableop_46_batch_normalization_7_moving_mean:G
9assignvariableop_47_batch_normalization_7_moving_variance:'
assignvariableop_48_adam_iter:	 )
assignvariableop_49_adam_beta_1: )
assignvariableop_50_adam_beta_2: (
assignvariableop_51_adam_decay: 0
&assignvariableop_52_adam_learning_rate: #
assignvariableop_53_total: #
assignvariableop_54_count: %
assignvariableop_55_total_1: %
assignvariableop_56_count_1: B
(assignvariableop_57_adam_conv2d_kernel_m: 4
&assignvariableop_58_adam_conv2d_bias_m: B
4assignvariableop_59_adam_batch_normalization_gamma_m: A
3assignvariableop_60_adam_batch_normalization_beta_m: D
*assignvariableop_61_adam_conv2d_1_kernel_m: @6
(assignvariableop_62_adam_conv2d_1_bias_m:@D
6assignvariableop_63_adam_batch_normalization_1_gamma_m:@C
5assignvariableop_64_adam_batch_normalization_1_beta_m:@D
*assignvariableop_65_adam_conv2d_2_kernel_m:@@6
(assignvariableop_66_adam_conv2d_2_bias_m:@D
6assignvariableop_67_adam_batch_normalization_2_gamma_m:@C
5assignvariableop_68_adam_batch_normalization_2_beta_m:@E
*assignvariableop_69_adam_conv2d_3_kernel_m:@?7
(assignvariableop_70_adam_conv2d_3_bias_m:	?E
6assignvariableop_71_adam_batch_normalization_3_gamma_m:	?D
5assignvariableop_72_adam_batch_normalization_3_beta_m:	?E
*assignvariableop_73_adam_conv2d_4_kernel_m:?@6
(assignvariableop_74_adam_conv2d_4_bias_m:@D
6assignvariableop_75_adam_batch_normalization_4_gamma_m:@C
5assignvariableop_76_adam_batch_normalization_4_beta_m:@D
*assignvariableop_77_adam_conv2d_5_kernel_m:@@6
(assignvariableop_78_adam_conv2d_5_bias_m:@D
6assignvariableop_79_adam_batch_normalization_5_gamma_m:@C
5assignvariableop_80_adam_batch_normalization_5_beta_m:@D
*assignvariableop_81_adam_conv2d_6_kernel_m:@ 6
(assignvariableop_82_adam_conv2d_6_bias_m: D
6assignvariableop_83_adam_batch_normalization_6_gamma_m: C
5assignvariableop_84_adam_batch_normalization_6_beta_m: D
*assignvariableop_85_adam_conv2d_7_kernel_m: 6
(assignvariableop_86_adam_conv2d_7_bias_m:D
6assignvariableop_87_adam_batch_normalization_7_gamma_m:C
5assignvariableop_88_adam_batch_normalization_7_beta_m:B
(assignvariableop_89_adam_conv2d_kernel_v: 4
&assignvariableop_90_adam_conv2d_bias_v: B
4assignvariableop_91_adam_batch_normalization_gamma_v: A
3assignvariableop_92_adam_batch_normalization_beta_v: D
*assignvariableop_93_adam_conv2d_1_kernel_v: @6
(assignvariableop_94_adam_conv2d_1_bias_v:@D
6assignvariableop_95_adam_batch_normalization_1_gamma_v:@C
5assignvariableop_96_adam_batch_normalization_1_beta_v:@D
*assignvariableop_97_adam_conv2d_2_kernel_v:@@6
(assignvariableop_98_adam_conv2d_2_bias_v:@D
6assignvariableop_99_adam_batch_normalization_2_gamma_v:@D
6assignvariableop_100_adam_batch_normalization_2_beta_v:@F
+assignvariableop_101_adam_conv2d_3_kernel_v:@?8
)assignvariableop_102_adam_conv2d_3_bias_v:	?F
7assignvariableop_103_adam_batch_normalization_3_gamma_v:	?E
6assignvariableop_104_adam_batch_normalization_3_beta_v:	?F
+assignvariableop_105_adam_conv2d_4_kernel_v:?@7
)assignvariableop_106_adam_conv2d_4_bias_v:@E
7assignvariableop_107_adam_batch_normalization_4_gamma_v:@D
6assignvariableop_108_adam_batch_normalization_4_beta_v:@E
+assignvariableop_109_adam_conv2d_5_kernel_v:@@7
)assignvariableop_110_adam_conv2d_5_bias_v:@E
7assignvariableop_111_adam_batch_normalization_5_gamma_v:@D
6assignvariableop_112_adam_batch_normalization_5_beta_v:@E
+assignvariableop_113_adam_conv2d_6_kernel_v:@ 7
)assignvariableop_114_adam_conv2d_6_bias_v: E
7assignvariableop_115_adam_batch_normalization_6_gamma_v: D
6assignvariableop_116_adam_batch_normalization_6_beta_v: E
+assignvariableop_117_adam_conv2d_7_kernel_v: 7
)assignvariableop_118_adam_conv2d_7_bias_v:E
7assignvariableop_119_adam_batch_normalization_7_gamma_v:D
6assignvariableop_120_adam_batch_normalization_7_beta_v:
identity_122??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_111?AssignVariableOp_112?AssignVariableOp_113?AssignVariableOp_114?AssignVariableOp_115?AssignVariableOp_116?AssignVariableOp_117?AssignVariableOp_118?AssignVariableOp_119?AssignVariableOp_12?AssignVariableOp_120?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?D
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:z*
dtype0*?C
value?CB?CzB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:z*
dtype0*?
value?B?zB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes~
|2z	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp,assignvariableop_2_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp+assignvariableop_3_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp2assignvariableop_4_batch_normalization_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp6assignvariableop_5_batch_normalization_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_1_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_1_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_2_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_2_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_2_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_2_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv2d_3_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv2d_3_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp/assignvariableop_20_batch_normalization_3_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp.assignvariableop_21_batch_normalization_3_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp5assignvariableop_22_batch_normalization_3_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp9assignvariableop_23_batch_normalization_3_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp#assignvariableop_24_conv2d_4_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp!assignvariableop_25_conv2d_4_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp/assignvariableop_26_batch_normalization_4_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp.assignvariableop_27_batch_normalization_4_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp5assignvariableop_28_batch_normalization_4_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp9assignvariableop_29_batch_normalization_4_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp#assignvariableop_30_conv2d_5_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp!assignvariableop_31_conv2d_5_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp/assignvariableop_32_batch_normalization_5_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp.assignvariableop_33_batch_normalization_5_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp5assignvariableop_34_batch_normalization_5_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp9assignvariableop_35_batch_normalization_5_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp#assignvariableop_36_conv2d_6_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp!assignvariableop_37_conv2d_6_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp/assignvariableop_38_batch_normalization_6_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp.assignvariableop_39_batch_normalization_6_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp5assignvariableop_40_batch_normalization_6_moving_meanIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp9assignvariableop_41_batch_normalization_6_moving_varianceIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp#assignvariableop_42_conv2d_7_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp!assignvariableop_43_conv2d_7_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp/assignvariableop_44_batch_normalization_7_gammaIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp.assignvariableop_45_batch_normalization_7_betaIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp5assignvariableop_46_batch_normalization_7_moving_meanIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp9assignvariableop_47_batch_normalization_7_moving_varianceIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOpassignvariableop_48_adam_iterIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOpassignvariableop_49_adam_beta_1Identity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOpassignvariableop_50_adam_beta_2Identity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOpassignvariableop_51_adam_decayIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp&assignvariableop_52_adam_learning_rateIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOpassignvariableop_53_totalIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOpassignvariableop_54_countIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOpassignvariableop_55_total_1Identity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOpassignvariableop_56_count_1Identity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp(assignvariableop_57_adam_conv2d_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp&assignvariableop_58_adam_conv2d_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp4assignvariableop_59_adam_batch_normalization_gamma_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp3assignvariableop_60_adam_batch_normalization_beta_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_conv2d_1_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_conv2d_1_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp6assignvariableop_63_adam_batch_normalization_1_gamma_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp5assignvariableop_64_adam_batch_normalization_1_beta_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_conv2d_2_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_conv2d_2_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp6assignvariableop_67_adam_batch_normalization_2_gamma_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp5assignvariableop_68_adam_batch_normalization_2_beta_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp*assignvariableop_69_adam_conv2d_3_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp(assignvariableop_70_adam_conv2d_3_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp6assignvariableop_71_adam_batch_normalization_3_gamma_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp5assignvariableop_72_adam_batch_normalization_3_beta_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp*assignvariableop_73_adam_conv2d_4_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp(assignvariableop_74_adam_conv2d_4_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp6assignvariableop_75_adam_batch_normalization_4_gamma_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp5assignvariableop_76_adam_batch_normalization_4_beta_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp*assignvariableop_77_adam_conv2d_5_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp(assignvariableop_78_adam_conv2d_5_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp6assignvariableop_79_adam_batch_normalization_5_gamma_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp5assignvariableop_80_adam_batch_normalization_5_beta_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp*assignvariableop_81_adam_conv2d_6_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp(assignvariableop_82_adam_conv2d_6_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp6assignvariableop_83_adam_batch_normalization_6_gamma_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp5assignvariableop_84_adam_batch_normalization_6_beta_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp*assignvariableop_85_adam_conv2d_7_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp(assignvariableop_86_adam_conv2d_7_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOp6assignvariableop_87_adam_batch_normalization_7_gamma_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOp5assignvariableop_88_adam_batch_normalization_7_beta_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOp(assignvariableop_89_adam_conv2d_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOp&assignvariableop_90_adam_conv2d_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp4assignvariableop_91_adam_batch_normalization_gamma_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp3assignvariableop_92_adam_batch_normalization_beta_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp*assignvariableop_93_adam_conv2d_1_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp(assignvariableop_94_adam_conv2d_1_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOp6assignvariableop_95_adam_batch_normalization_1_gamma_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOp5assignvariableop_96_adam_batch_normalization_1_beta_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97?
AssignVariableOp_97AssignVariableOp*assignvariableop_97_adam_conv2d_2_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98?
AssignVariableOp_98AssignVariableOp(assignvariableop_98_adam_conv2d_2_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99?
AssignVariableOp_99AssignVariableOp6assignvariableop_99_adam_batch_normalization_2_gamma_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100?
AssignVariableOp_100AssignVariableOp6assignvariableop_100_adam_batch_normalization_2_beta_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101?
AssignVariableOp_101AssignVariableOp+assignvariableop_101_adam_conv2d_3_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102?
AssignVariableOp_102AssignVariableOp)assignvariableop_102_adam_conv2d_3_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103?
AssignVariableOp_103AssignVariableOp7assignvariableop_103_adam_batch_normalization_3_gamma_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104?
AssignVariableOp_104AssignVariableOp6assignvariableop_104_adam_batch_normalization_3_beta_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105?
AssignVariableOp_105AssignVariableOp+assignvariableop_105_adam_conv2d_4_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106?
AssignVariableOp_106AssignVariableOp)assignvariableop_106_adam_conv2d_4_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107?
AssignVariableOp_107AssignVariableOp7assignvariableop_107_adam_batch_normalization_4_gamma_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108?
AssignVariableOp_108AssignVariableOp6assignvariableop_108_adam_batch_normalization_4_beta_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109?
AssignVariableOp_109AssignVariableOp+assignvariableop_109_adam_conv2d_5_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110?
AssignVariableOp_110AssignVariableOp)assignvariableop_110_adam_conv2d_5_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111?
AssignVariableOp_111AssignVariableOp7assignvariableop_111_adam_batch_normalization_5_gamma_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112?
AssignVariableOp_112AssignVariableOp6assignvariableop_112_adam_batch_normalization_5_beta_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113?
AssignVariableOp_113AssignVariableOp+assignvariableop_113_adam_conv2d_6_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114?
AssignVariableOp_114AssignVariableOp)assignvariableop_114_adam_conv2d_6_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115?
AssignVariableOp_115AssignVariableOp7assignvariableop_115_adam_batch_normalization_6_gamma_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116?
AssignVariableOp_116AssignVariableOp6assignvariableop_116_adam_batch_normalization_6_beta_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117?
AssignVariableOp_117AssignVariableOp+assignvariableop_117_adam_conv2d_7_kernel_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118?
AssignVariableOp_118AssignVariableOp)assignvariableop_118_adam_conv2d_7_bias_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119?
AssignVariableOp_119AssignVariableOp7assignvariableop_119_adam_batch_normalization_7_gamma_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120?
AssignVariableOp_120AssignVariableOp6assignvariableop_120_adam_batch_normalization_7_beta_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1209
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_121Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_121i
Identity_122IdentityIdentity_121:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_122?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"%
identity_122Identity_122:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202*
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
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
C__inference_conv2d_7_layer_call_and_return_conditional_losses_10798

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10622

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
B__inference_conv2d_2_layer_call_and_return_conditional_losses_6499

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
5__inference_batch_normalization_3_layer_call_fn_10076

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_65832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????  ?: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6048

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
b
F__inference_activation_1_layer_call_and_return_conditional_losses_6476

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????@2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
C__inference_conv2d_6_layer_call_and_return_conditional_losses_10595

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
G
+__inference_activation_1_layer_call_fn_9733

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_64762
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_7095
input_1!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?%

unknown_23:?@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:@

unknown_28:@$

unknown_29:@@

unknown_30:@

unknown_31:@

unknown_32:@

unknown_33:@

unknown_34:@$

unknown_35:@ 

unknown_36: 

unknown_37: 

unknown_38: 

unknown_39: 

unknown_40: $

unknown_41: 

unknown_42:

unknown_43:

unknown_44:

unknown_45:

unknown_46:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_69962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes
}:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
$__inference_model_layer_call_fn_9397

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?%

unknown_23:?@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:@

unknown_28:@$

unknown_29:@@

unknown_30:@

unknown_31:@

unknown_32:@

unknown_33:@

unknown_34:@$

unknown_35:@ 

unknown_36: 

unknown_37: 

unknown_38: 

unknown_39: 

unknown_40: $

unknown_41: 

unknown_42:

unknown_43:

unknown_44:

unknown_45:

unknown_46:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*B
_read_only_resource_inputs$
" 	
 !"%&'(+,-.*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_79342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes
}:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_3_layer_call_fn_10063

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_57962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_10001

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?

S__inference_max_pooling_with_argmax2d_layer_call_and_return_conditional_losses_6425

inputs
identity

identity_1?
MaxPoolWithArgmaxMaxPoolWithArgmaxinputs*
T0*N
_output_shapes<
::??????????? :??????????? *
ksize
*
paddingSAME*
strides
2
MaxPoolWithArgmax{
CastCastMaxPoolWithArgmax:argmax:0*

DstT0*

SrcT0	*1
_output_shapes
:??????????? 2
Castx
IdentityIdentityMaxPoolWithArgmax:output:0*
T0*1
_output_shapes
:??????????? 2

Identityj

Identity_1IdentityCast:y:0*
T0*1
_output_shapes
:??????????? 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_10473

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9854

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_10234

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
4__inference_batch_normalization_2_layer_call_fn_9880

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_56702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
%__inference_conv2d_layer_call_fn_9416

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_63772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
U__inference_max_pooling_with_argmax2d_1_layer_call_and_return_conditional_losses_7547

inputs
identity

identity_1?
MaxPoolWithArgmaxMaxPoolWithArgmaxinputs*
T0*J
_output_shapes8
6:?????????@@@:?????????@@@*
ksize
*
paddingSAME*
strides
2
MaxPoolWithArgmaxy
CastCastMaxPoolWithArgmax:argmax:0*

DstT0*

SrcT0	*/
_output_shapes
:?????????@@@2
Castv
IdentityIdentityMaxPoolWithArgmax:output:0*
T0*/
_output_shapes
:?????????@@@2

Identityh

Identity_1IdentityCast:y:0*
T0*/
_output_shapes
:?????????@@@2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10861

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
5__inference_batch_normalization_7_layer_call_fn_10918

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_69782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?+
s
I__inference_max_unpooling2d_layer_call_and_return_conditional_losses_6655

inputs
inputs_1
identityh
CastCastinputs_1*

DstT0*

SrcT0*0
_output_shapes
:??????????2
CastD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2T
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapem
ReshapeReshapeCast:y:0Reshape/shape:output:0*
T0*#
_output_shapes
:?????????2	
Reshapek
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim?

ExpandDims
ExpandDimsReshape:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2

ExpandDimsu
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_1/shapeq
	Reshape_1ReshapeinputsReshape_1/shape:output:0*
T0*#
_output_shapes
:?????????2
	Reshape_1?
Rank/packedPackstrided_slice:output:0mul:z:0	mul_1:z:0strided_slice_3:output:0*
N*
T0*
_output_shapes
:2
Rank/packedN
RankConst*
_output_shapes
: *
dtype0*
value	B :2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltan
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:2
range?

Prod/inputPackstrided_slice:output:0mul:z:0	mul_1:z:0strided_slice_3:output:0*
N*
T0*
_output_shapes
:2

Prod/inputZ
ProdProdProd/input:output:0range:output:0*
T0*
_output_shapes
: 2
Prodg
ScatterNd/shapePackProd:output:0*
N*
T0*
_output_shapes
:2
ScatterNd/shape?
	ScatterNd	ScatterNdExpandDims:output:0Reshape_1:output:0ScatterNd/shape:output:0*
T0*
Tindices0*#
_output_shapes
:?????????2
	ScatterNd{
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????        ?   2
Reshape_2/shape?
	Reshape_2ReshapeScatterNd:output:0Reshape_2/shape:output:0*
T0*0
_output_shapes
:?????????  ?2
	Reshape_2o
IdentityIdentityReshape_2:output:0*
T0*0
_output_shapes
:?????????  ?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:??????????:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs:XT
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
B__inference_conv2d_3_layer_call_and_return_conditional_losses_9956

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?2	
BiasAddt
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
c
G__inference_activation_3_layer_call_and_return_conditional_losses_10094

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????  ?2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????  ?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?+
u
K__inference_max_unpooling2d_1_layer_call_and_return_conditional_losses_6751

inputs
inputs_1
identityg
CastCastinputs_1*

DstT0*

SrcT0*/
_output_shapes
:?????????  @2
CastD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2T
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapem
ReshapeReshapeCast:y:0Reshape/shape:output:0*
T0*#
_output_shapes
:?????????2	
Reshapek
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim?

ExpandDims
ExpandDimsReshape:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2

ExpandDimsu
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_1/shapeq
	Reshape_1ReshapeinputsReshape_1/shape:output:0*
T0*#
_output_shapes
:?????????2
	Reshape_1?
Rank/packedPackstrided_slice:output:0mul:z:0	mul_1:z:0strided_slice_3:output:0*
N*
T0*
_output_shapes
:2
Rank/packedN
RankConst*
_output_shapes
: *
dtype0*
value	B :2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltan
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:2
range?

Prod/inputPackstrided_slice:output:0mul:z:0	mul_1:z:0strided_slice_3:output:0*
N*
T0*
_output_shapes
:2

Prod/inputZ
ProdProdProd/input:output:0range:output:0*
T0*
_output_shapes
: 2
Prodg
ScatterNd/shapePackProd:output:0*
N*
T0*
_output_shapes
:2
ScatterNd/shape?
	ScatterNd	ScatterNdExpandDims:output:0Reshape_1:output:0ScatterNd/shape:output:0*
T0*
Tindices0*#
_output_shapes
:?????????2
	ScatterNd{
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????@   @   @   2
Reshape_2/shape?
	Reshape_2ReshapeScatterNd:output:0Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????@@@2
	Reshape_2n
IdentityIdentityReshape_2:output:0*
T0*/
_output_shapes
:?????????@@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????  @:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
c
G__inference_activation_5_layer_call_and_return_conditional_losses_10530

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@@:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_6_layer_call_fn_10702

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_61742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10825

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_7134

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
d
8__inference_max_pooling_with_argmax2d_layer_call_fn_9580

inputs
identity

identity_1?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::??????????? :??????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_max_pooling_with_argmax2d_layer_call_and_return_conditional_losses_76302
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:??????????? 2

Identityz

Identity_1IdentityPartitionedCall:output:1*
T0*1
_output_shapes
:??????????? 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_5374

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_7425

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????  ?: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?+
x
L__inference_max_unpooling2d_1_layer_call_and_return_conditional_losses_10376
inputs_0
inputs_1
identityg
CastCastinputs_1*

DstT0*

SrcT0*/
_output_shapes
:?????????  @2
CastF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2T
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapem
ReshapeReshapeCast:y:0Reshape/shape:output:0*
T0*#
_output_shapes
:?????????2	
Reshapek
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim?

ExpandDims
ExpandDimsReshape:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2

ExpandDimsu
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_1/shapes
	Reshape_1Reshapeinputs_0Reshape_1/shape:output:0*
T0*#
_output_shapes
:?????????2
	Reshape_1?
Rank/packedPackstrided_slice:output:0mul:z:0	mul_1:z:0strided_slice_3:output:0*
N*
T0*
_output_shapes
:2
Rank/packedN
RankConst*
_output_shapes
: *
dtype0*
value	B :2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltan
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:2
range?

Prod/inputPackstrided_slice:output:0mul:z:0	mul_1:z:0strided_slice_3:output:0*
N*
T0*
_output_shapes
:2

Prod/inputZ
ProdProdProd/input:output:0range:output:0*
T0*
_output_shapes
: 2
Prodg
ScatterNd/shapePackProd:output:0*
N*
T0*
_output_shapes
:2
ScatterNd/shape?
	ScatterNd	ScatterNdExpandDims:output:0Reshape_1:output:0ScatterNd/shape:output:0*
T0*
Tindices0*#
_output_shapes
:?????????2
	ScatterNd{
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????@   @   @   2
Reshape_2/shape?
	Reshape_2ReshapeScatterNd:output:0Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????@@@2
	Reshape_2n
IdentityIdentityReshape_2:output:0*
T0*/
_output_shapes
:?????????@@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????  @:?????????  @:Y U
/
_output_shapes
:?????????  @
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????  @
"
_user_specified_name
inputs/1
?

S__inference_max_pooling_with_argmax2d_layer_call_and_return_conditional_losses_9558

inputs
identity

identity_1?
MaxPoolWithArgmaxMaxPoolWithArgmaxinputs*
T0*N
_output_shapes<
::??????????? :??????????? *
ksize
*
paddingSAME*
strides
2
MaxPoolWithArgmax{
CastCastMaxPoolWithArgmax:argmax:0*

DstT0*

SrcT0	*1
_output_shapes
:??????????? 2
Castx
IdentityIdentityMaxPoolWithArgmax:output:0*
T0*1
_output_shapes
:??????????? 2

Identityj

Identity_1IdentityCast:y:0*
T0*1
_output_shapes
:??????????? 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10879

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
4__inference_batch_normalization_1_layer_call_fn_9684

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_55002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5544

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
U__inference_max_pooling_with_argmax2d_3_layer_call_and_return_conditional_losses_7381

inputs
identity

identity_1?
MaxPoolWithArgmaxMaxPoolWithArgmaxinputs*
T0*L
_output_shapes:
8:??????????:??????????*
ksize
*
paddingSAME*
strides
2
MaxPoolWithArgmaxz
CastCastMaxPoolWithArgmax:argmax:0*

DstT0*

SrcT0	*0
_output_shapes
:??????????2
Castw
IdentityIdentityMaxPoolWithArgmax:output:0*
T0*0
_output_shapes
:??????????2

Identityi

Identity_1IdentityCast:y:0*
T0*0
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9653

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
@__inference_conv2d_layer_call_and_return_conditional_losses_6377

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9818

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
U__inference_max_pooling_with_argmax2d_1_layer_call_and_return_conditional_losses_9749

inputs
identity

identity_1?
MaxPoolWithArgmaxMaxPoolWithArgmaxinputs*
T0*J
_output_shapes8
6:?????????@@@:?????????@@@*
ksize
*
paddingSAME*
strides
2
MaxPoolWithArgmaxy
CastCastMaxPoolWithArgmax:argmax:0*

DstT0*

SrcT0	*/
_output_shapes
:?????????@@@2
Castv
IdentityIdentityMaxPoolWithArgmax:output:0*
T0*/
_output_shapes
:?????????@@@2

Identityh

Identity_1IdentityCast:y:0*
T0*/
_output_shapes
:?????????@@@2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_10019

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????  ?: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
g
;__inference_max_pooling_with_argmax2d_3_layer_call_fn_10122

inputs
identity

identity_1?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *L
_output_shapes:
8:??????????:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_max_pooling_with_argmax2d_3_layer_call_and_return_conditional_losses_66082
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identityy

Identity_1IdentityPartitionedCall:output:1*
T0*0
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5752

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_7_layer_call_fn_10892

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_62562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
H
,__inference_activation_5_layer_call_fn_10535

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_5_layer_call_and_return_conditional_losses_68012
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@@:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?	
?
2__inference_batch_normalization_layer_call_fn_9514

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_54182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_6174

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
b
F__inference_activation_3_layer_call_and_return_conditional_losses_6598

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????  ?2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????  ?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
f
:__inference_max_pooling_with_argmax2d_2_layer_call_fn_9939

inputs
identity

identity_1?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:?????????  @:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_max_pooling_with_argmax2d_2_layer_call_and_return_conditional_losses_65472
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????  @2

Identityx

Identity_1IdentityPartitionedCall:output:1*
T0*/
_output_shapes
:?????????  @2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@@:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
U__inference_max_pooling_with_argmax2d_1_layer_call_and_return_conditional_losses_6486

inputs
identity

identity_1?
MaxPoolWithArgmaxMaxPoolWithArgmaxinputs*
T0*J
_output_shapes8
6:?????????@@@:?????????@@@*
ksize
*
paddingSAME*
strides
2
MaxPoolWithArgmaxy
CastCastMaxPoolWithArgmax:argmax:0*

DstT0*

SrcT0	*/
_output_shapes
:?????????@@@2
Castv
IdentityIdentityMaxPoolWithArgmax:output:0*
T0*/
_output_shapes
:?????????@@@2

Identityh

Identity_1IdentityCast:y:0*
T0*/
_output_shapes
:?????????@@@2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
B__inference_conv2d_1_layer_call_and_return_conditional_losses_6438

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_2_layer_call_fn_9893

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_65222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_9470

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?

S__inference_max_pooling_with_argmax2d_layer_call_and_return_conditional_losses_9566

inputs
identity

identity_1?
MaxPoolWithArgmaxMaxPoolWithArgmaxinputs*
T0*N
_output_shapes<
::??????????? :??????????? *
ksize
*
paddingSAME*
strides
2
MaxPoolWithArgmax{
CastCastMaxPoolWithArgmax:argmax:0*

DstT0*

SrcT0	*1
_output_shapes
:??????????? 2
Castx
IdentityIdentityMaxPoolWithArgmax:output:0*
T0*1
_output_shapes
:??????????? 2

Identityj

Identity_1IdentityCast:y:0*
T0*1
_output_shapes
:??????????? 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6300

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5670

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_7674

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
B__inference_conv2d_6_layer_call_and_return_conditional_losses_6859

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
[
/__inference_max_unpooling2d_layer_call_fn_10179
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_unpooling2d_layer_call_and_return_conditional_losses_66552
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????  ?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:??????????:??????????:Z V
0
_output_shapes
:??????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?
?
V__inference_max_pooling_with_argmax2d_3_layer_call_and_return_conditional_losses_10115

inputs
identity

identity_1?
MaxPoolWithArgmaxMaxPoolWithArgmaxinputs*
T0*L
_output_shapes:
8:??????????:??????????*
ksize
*
paddingSAME*
strides
2
MaxPoolWithArgmaxz
CastCastMaxPoolWithArgmax:argmax:0*

DstT0*

SrcT0	*0
_output_shapes
:??????????2
Castw
IdentityIdentityMaxPoolWithArgmax:output:0*
T0*0
_output_shapes
:??????????2

Identityi

Identity_1IdentityCast:y:0*
T0*0
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
f
:__inference_max_pooling_with_argmax2d_1_layer_call_fn_9756

inputs
identity

identity_1?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:?????????@@@:?????????@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_max_pooling_with_argmax2d_1_layer_call_and_return_conditional_losses_64862
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@@@2

Identityx

Identity_1IdentityPartitionedCall:output:1*
T0*/
_output_shapes
:?????????@@@2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
5__inference_batch_normalization_4_layer_call_fn_10322

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_73352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
U__inference_max_pooling_with_argmax2d_3_layer_call_and_return_conditional_losses_6608

inputs
identity

identity_1?
MaxPoolWithArgmaxMaxPoolWithArgmaxinputs*
T0*L
_output_shapes:
8:??????????:??????????*
ksize
*
paddingSAME*
strides
2
MaxPoolWithArgmaxz
CastCastMaxPoolWithArgmax:argmax:0*

DstT0*

SrcT0	*0
_output_shapes
:??????????2
Castw
IdentityIdentityMaxPoolWithArgmax:output:0*
T0*0
_output_shapes
:??????????2

Identityi

Identity_1IdentityCast:y:0*
T0*0
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
??
?
?__inference_model_layer_call_and_return_conditional_losses_8270
input_1%
conv2d_8137: 
conv2d_8139: &
batch_normalization_8142: &
batch_normalization_8144: &
batch_normalization_8146: &
batch_normalization_8148: '
conv2d_1_8154: @
conv2d_1_8156:@(
batch_normalization_1_8159:@(
batch_normalization_1_8161:@(
batch_normalization_1_8163:@(
batch_normalization_1_8165:@'
conv2d_2_8171:@@
conv2d_2_8173:@(
batch_normalization_2_8176:@(
batch_normalization_2_8178:@(
batch_normalization_2_8180:@(
batch_normalization_2_8182:@(
conv2d_3_8188:@?
conv2d_3_8190:	?)
batch_normalization_3_8193:	?)
batch_normalization_3_8195:	?)
batch_normalization_3_8197:	?)
batch_normalization_3_8199:	?(
conv2d_4_8206:?@
conv2d_4_8208:@(
batch_normalization_4_8211:@(
batch_normalization_4_8213:@(
batch_normalization_4_8215:@(
batch_normalization_4_8217:@'
conv2d_5_8222:@@
conv2d_5_8224:@(
batch_normalization_5_8227:@(
batch_normalization_5_8229:@(
batch_normalization_5_8231:@(
batch_normalization_5_8233:@'
conv2d_6_8238:@ 
conv2d_6_8240: (
batch_normalization_6_8243: (
batch_normalization_6_8245: (
batch_normalization_6_8247: (
batch_normalization_6_8249: '
conv2d_7_8254: 
conv2d_7_8256:(
batch_normalization_7_8259:(
batch_normalization_7_8261:(
batch_normalization_7_8263:(
batch_normalization_7_8265:
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?-batch_normalization_5/StatefulPartitionedCall?-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_8137conv2d_8139*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_63772 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_8142batch_normalization_8144batch_normalization_8146batch_normalization_8148*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_64002-
+batch_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_64152
activation/PartitionedCall?
)max_pooling_with_argmax2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::??????????? :??????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_max_pooling_with_argmax2d_layer_call_and_return_conditional_losses_64252+
)max_pooling_with_argmax2d/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall2max_pooling_with_argmax2d/PartitionedCall:output:0conv2d_1_8154conv2d_1_8156*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_64382"
 conv2d_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_8159batch_normalization_1_8161batch_normalization_1_8163batch_normalization_1_8165*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_64612/
-batch_normalization_1/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_64762
activation_1/PartitionedCall?
+max_pooling_with_argmax2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:?????????@@@:?????????@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_max_pooling_with_argmax2d_1_layer_call_and_return_conditional_losses_64862-
+max_pooling_with_argmax2d_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall4max_pooling_with_argmax2d_1/PartitionedCall:output:0conv2d_2_8171conv2d_2_8173*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_64992"
 conv2d_2/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_8176batch_normalization_2_8178batch_normalization_2_8180batch_normalization_2_8182*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_65222/
-batch_normalization_2/StatefulPartitionedCall?
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_65372
activation_2/PartitionedCall?
+max_pooling_with_argmax2d_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:?????????  @:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_max_pooling_with_argmax2d_2_layer_call_and_return_conditional_losses_65472-
+max_pooling_with_argmax2d_2/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall4max_pooling_with_argmax2d_2/PartitionedCall:output:0conv2d_3_8188conv2d_3_8190*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_65602"
 conv2d_3/StatefulPartitionedCall?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_8193batch_normalization_3_8195batch_normalization_3_8197batch_normalization_3_8199*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_65832/
-batch_normalization_3/StatefulPartitionedCall?
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_65982
activation_3/PartitionedCall?
+max_pooling_with_argmax2d_3/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *L
_output_shapes:
8:??????????:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_max_pooling_with_argmax2d_3_layer_call_and_return_conditional_losses_66082-
+max_pooling_with_argmax2d_3/PartitionedCall?
max_unpooling2d/PartitionedCallPartitionedCall4max_pooling_with_argmax2d_3/PartitionedCall:output:04max_pooling_with_argmax2d_3/PartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_unpooling2d_layer_call_and_return_conditional_losses_66552!
max_unpooling2d/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_unpooling2d/PartitionedCall:output:0conv2d_4_8206conv2d_4_8208*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_66672"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_8211batch_normalization_4_8213batch_normalization_4_8215batch_normalization_4_8217*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_66902/
-batch_normalization_4/StatefulPartitionedCall?
activation_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_67052
activation_4/PartitionedCall?
!max_unpooling2d_1/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:04max_pooling_with_argmax2d_2/PartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_unpooling2d_1_layer_call_and_return_conditional_losses_67512#
!max_unpooling2d_1/PartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall*max_unpooling2d_1/PartitionedCall:output:0conv2d_5_8222conv2d_5_8224*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_67632"
 conv2d_5/StatefulPartitionedCall?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_8227batch_normalization_5_8229batch_normalization_5_8231batch_normalization_5_8233*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_67862/
-batch_normalization_5/StatefulPartitionedCall?
activation_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_5_layer_call_and_return_conditional_losses_68012
activation_5/PartitionedCall?
!max_unpooling2d_2/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:04max_pooling_with_argmax2d_1/PartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_unpooling2d_2_layer_call_and_return_conditional_losses_68472#
!max_unpooling2d_2/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall*max_unpooling2d_2/PartitionedCall:output:0conv2d_6_8238conv2d_6_8240*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_68592"
 conv2d_6/StatefulPartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_8243batch_normalization_6_8245batch_normalization_6_8247batch_normalization_6_8249*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_68822/
-batch_normalization_6/StatefulPartitionedCall?
activation_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_6_layer_call_and_return_conditional_losses_68972
activation_6/PartitionedCall?
!max_unpooling2d_3/PartitionedCallPartitionedCall%activation_6/PartitionedCall:output:02max_pooling_with_argmax2d/PartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_unpooling2d_3_layer_call_and_return_conditional_losses_69432#
!max_unpooling2d_3/PartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall*max_unpooling2d_3/PartitionedCall:output:0conv2d_7_8254conv2d_7_8256*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_7_layer_call_and_return_conditional_losses_69552"
 conv2d_7/StatefulPartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_7_8259batch_normalization_7_8261batch_normalization_7_8263batch_normalization_7_8265*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_69782/
-batch_normalization_7/StatefulPartitionedCall?
activation_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_7_layer_call_and_return_conditional_losses_69932
activation_7/PartitionedCall?
IdentityIdentity%activation_7/PartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes
}:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6786

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?

S__inference_max_pooling_with_argmax2d_layer_call_and_return_conditional_losses_7630

inputs
identity

identity_1?
MaxPoolWithArgmaxMaxPoolWithArgmaxinputs*
T0*N
_output_shapes<
::??????????? :??????????? *
ksize
*
paddingSAME*
strides
2
MaxPoolWithArgmax{
CastCastMaxPoolWithArgmax:argmax:0*

DstT0*

SrcT0	*1
_output_shapes
:??????????? 2
Castx
IdentityIdentityMaxPoolWithArgmax:output:0*
T0*1
_output_shapes
:??????????? 2

Identityj

Identity_1IdentityCast:y:0*
T0*1
_output_shapes
:??????????? 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
G
+__inference_activation_2_layer_call_fn_9916

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_65372
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@@:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?+
x
L__inference_max_unpooling2d_2_layer_call_and_return_conditional_losses_10579
inputs_0
inputs_1
identityg
CastCastinputs_1*

DstT0*

SrcT0*/
_output_shapes
:?????????@@@2
CastF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2T
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapem
ReshapeReshapeCast:y:0Reshape/shape:output:0*
T0*#
_output_shapes
:?????????2	
Reshapek
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim?

ExpandDims
ExpandDimsReshape:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2

ExpandDimsu
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_1/shapes
	Reshape_1Reshapeinputs_0Reshape_1/shape:output:0*
T0*#
_output_shapes
:?????????2
	Reshape_1?
Rank/packedPackstrided_slice:output:0mul:z:0	mul_1:z:0strided_slice_3:output:0*
N*
T0*
_output_shapes
:2
Rank/packedN
RankConst*
_output_shapes
: *
dtype0*
value	B :2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltan
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:2
range?

Prod/inputPackstrided_slice:output:0mul:z:0	mul_1:z:0strided_slice_3:output:0*
N*
T0*
_output_shapes
:2

Prod/inputZ
ProdProdProd/input:output:0range:output:0*
T0*
_output_shapes
: 2
Prodg
ScatterNd/shapePackProd:output:0*
N*
T0*
_output_shapes
:2
ScatterNd/shape?
	ScatterNd	ScatterNdExpandDims:output:0Reshape_1:output:0ScatterNd/shape:output:0*
T0*
Tindices0*#
_output_shapes
:?????????2
	ScatterNd{
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"?????   ?   @   2
Reshape_2/shape?
	Reshape_2ReshapeScatterNd:output:0Reshape_2/shape:output:0*
T0*1
_output_shapes
:???????????@2
	Reshape_2p
IdentityIdentityReshape_2:output:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????@@@:?????????@@@:Y U
/
_output_shapes
:?????????@@@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????@@@
"
_user_specified_name
inputs/1
?
?
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_6256

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_activation_6_layer_call_and_return_conditional_losses_6897

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:??????????? 2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
5__inference_batch_normalization_6_layer_call_fn_10728

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_72012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
b
F__inference_activation_2_layer_call_and_return_conditional_losses_6537

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????@@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@@:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
E
)__inference_activation_layer_call_fn_9550

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_64152
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9800

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5796

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_6004

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
5__inference_batch_normalization_7_layer_call_fn_10931

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_71342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
F__inference_activation_4_layer_call_and_return_conditional_losses_6705

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????  @2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10640

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6583

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????  ?: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9671

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?+
v
J__inference_max_unpooling2d_layer_call_and_return_conditional_losses_10173
inputs_0
inputs_1
identityh
CastCastinputs_1*

DstT0*

SrcT0*0
_output_shapes
:??????????2
CastF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2T
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapem
ReshapeReshapeCast:y:0Reshape/shape:output:0*
T0*#
_output_shapes
:?????????2	
Reshapek
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim?

ExpandDims
ExpandDimsReshape:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2

ExpandDimsu
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_1/shapes
	Reshape_1Reshapeinputs_0Reshape_1/shape:output:0*
T0*#
_output_shapes
:?????????2
	Reshape_1?
Rank/packedPackstrided_slice:output:0mul:z:0	mul_1:z:0strided_slice_3:output:0*
N*
T0*
_output_shapes
:2
Rank/packedN
RankConst*
_output_shapes
: *
dtype0*
value	B :2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltan
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:2
range?

Prod/inputPackstrided_slice:output:0mul:z:0	mul_1:z:0strided_slice_3:output:0*
N*
T0*
_output_shapes
:2

Prod/inputZ
ProdProdProd/input:output:0range:output:0*
T0*
_output_shapes
: 2
Prodg
ScatterNd/shapePackProd:output:0*
N*
T0*
_output_shapes
:2
ScatterNd/shape?
	ScatterNd	ScatterNdExpandDims:output:0Reshape_1:output:0ScatterNd/shape:output:0*
T0*
Tindices0*#
_output_shapes
:?????????2
	ScatterNd{
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????        ?   2
Reshape_2/shape?
	Reshape_2ReshapeScatterNd:output:0Reshape_2/shape:output:0*
T0*0
_output_shapes
:?????????  ?2
	Reshape_2o
IdentityIdentityReshape_2:output:0*
T0*0
_output_shapes
:?????????  ?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:??????????:??????????:Z V
0
_output_shapes
:??????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?
?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_9452

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
f
:__inference_max_pooling_with_argmax2d_2_layer_call_fn_9946

inputs
identity

identity_1?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:?????????  @:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_max_pooling_with_argmax2d_2_layer_call_and_return_conditional_losses_74642
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????  @2

Identityx

Identity_1IdentityPartitionedCall:output:1*
T0*/
_output_shapes
:?????????  @2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@@:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
'__inference_conv2d_1_layer_call_fn_9599

inputs!
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_64382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
H
,__inference_activation_6_layer_call_fn_10738

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_6_layer_call_and_return_conditional_losses_68972
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
2__inference_batch_normalization_layer_call_fn_9527

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_64002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
]
1__inference_max_unpooling2d_3_layer_call_fn_10788
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_unpooling2d_3_layer_call_and_return_conditional_losses_69432
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::??????????? :??????????? :[ W
1
_output_shapes
:??????????? 
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:??????????? 
"
_user_specified_name
inputs/1
?	
?
5__inference_batch_normalization_7_layer_call_fn_10905

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_63002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?+
x
L__inference_max_unpooling2d_3_layer_call_and_return_conditional_losses_10782
inputs_0
inputs_1
identityi
CastCastinputs_1*

DstT0*

SrcT0*1
_output_shapes
:??????????? 2
CastF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2T
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapem
ReshapeReshapeCast:y:0Reshape/shape:output:0*
T0*#
_output_shapes
:?????????2	
Reshapek
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim?

ExpandDims
ExpandDimsReshape:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2

ExpandDimsu
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape_1/shapes
	Reshape_1Reshapeinputs_0Reshape_1/shape:output:0*
T0*#
_output_shapes
:?????????2
	Reshape_1?
Rank/packedPackstrided_slice:output:0mul:z:0	mul_1:z:0strided_slice_3:output:0*
N*
T0*
_output_shapes
:2
Rank/packedN
RankConst*
_output_shapes
: *
dtype0*
value	B :2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltan
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:2
range?

Prod/inputPackstrided_slice:output:0mul:z:0	mul_1:z:0strided_slice_3:output:0*
N*
T0*
_output_shapes
:2

Prod/inputZ
ProdProdProd/input:output:0range:output:0*
T0*
_output_shapes
: 2
Prodg
ScatterNd/shapePackProd:output:0*
N*
T0*
_output_shapes
:2
ScatterNd/shape?
	ScatterNd	ScatterNdExpandDims:output:0Reshape_1:output:0ScatterNd/shape:output:0*
T0*
Tindices0*#
_output_shapes
:?????????2
	ScatterNd{
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????          2
Reshape_2/shape?
	Reshape_2ReshapeScatterNd:output:0Reshape_2/shape:output:0*
T0*1
_output_shapes
:??????????? 2
	Reshape_2p
IdentityIdentityReshape_2:output:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::??????????? :??????????? :[ W
1
_output_shapes
:??????????? 
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:??????????? 
"
_user_specified_name
inputs/1
?
?
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10658

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_4_layer_call_fn_10296

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_59222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
??
?7
__inference__traced_save_11327
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_3_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_m_read_readvariableop5
1savev2_adam_conv2d_4_kernel_m_read_readvariableop3
/savev2_adam_conv2d_4_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_m_read_readvariableop5
1savev2_adam_conv2d_5_kernel_m_read_readvariableop3
/savev2_adam_conv2d_5_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_5_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_5_beta_m_read_readvariableop5
1savev2_adam_conv2d_6_kernel_m_read_readvariableop3
/savev2_adam_conv2d_6_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_6_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_6_beta_m_read_readvariableop5
1savev2_adam_conv2d_7_kernel_m_read_readvariableop3
/savev2_adam_conv2d_7_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_7_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_7_beta_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_v_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop3
/savev2_adam_conv2d_3_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_v_read_readvariableop5
1savev2_adam_conv2d_4_kernel_v_read_readvariableop3
/savev2_adam_conv2d_4_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_v_read_readvariableop5
1savev2_adam_conv2d_5_kernel_v_read_readvariableop3
/savev2_adam_conv2d_5_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_5_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_5_beta_v_read_readvariableop5
1savev2_adam_conv2d_6_kernel_v_read_readvariableop3
/savev2_adam_conv2d_6_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_6_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_6_beta_v_read_readvariableop5
1savev2_adam_conv2d_7_kernel_v_read_readvariableop3
/savev2_adam_conv2d_7_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_7_beta_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?D
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:z*
dtype0*?C
value?CB?CzB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:z*
dtype0*?
value?B?zB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?5
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop<savev2_adam_batch_normalization_2_beta_m_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop<savev2_adam_batch_normalization_3_beta_m_read_readvariableop1savev2_adam_conv2d_4_kernel_m_read_readvariableop/savev2_adam_conv2d_4_bias_m_read_readvariableop=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop<savev2_adam_batch_normalization_4_beta_m_read_readvariableop1savev2_adam_conv2d_5_kernel_m_read_readvariableop/savev2_adam_conv2d_5_bias_m_read_readvariableop=savev2_adam_batch_normalization_5_gamma_m_read_readvariableop<savev2_adam_batch_normalization_5_beta_m_read_readvariableop1savev2_adam_conv2d_6_kernel_m_read_readvariableop/savev2_adam_conv2d_6_bias_m_read_readvariableop=savev2_adam_batch_normalization_6_gamma_m_read_readvariableop<savev2_adam_batch_normalization_6_beta_m_read_readvariableop1savev2_adam_conv2d_7_kernel_m_read_readvariableop/savev2_adam_conv2d_7_bias_m_read_readvariableop=savev2_adam_batch_normalization_7_gamma_m_read_readvariableop<savev2_adam_batch_normalization_7_beta_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop<savev2_adam_batch_normalization_2_beta_v_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop<savev2_adam_batch_normalization_3_beta_v_read_readvariableop1savev2_adam_conv2d_4_kernel_v_read_readvariableop/savev2_adam_conv2d_4_bias_v_read_readvariableop=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop<savev2_adam_batch_normalization_4_beta_v_read_readvariableop1savev2_adam_conv2d_5_kernel_v_read_readvariableop/savev2_adam_conv2d_5_bias_v_read_readvariableop=savev2_adam_batch_normalization_5_gamma_v_read_readvariableop<savev2_adam_batch_normalization_5_beta_v_read_readvariableop1savev2_adam_conv2d_6_kernel_v_read_readvariableop/savev2_adam_conv2d_6_bias_v_read_readvariableop=savev2_adam_batch_normalization_6_gamma_v_read_readvariableop<savev2_adam_batch_normalization_6_beta_v_read_readvariableop1savev2_adam_conv2d_7_kernel_v_read_readvariableop/savev2_adam_conv2d_7_bias_v_read_readvariableop=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop<savev2_adam_batch_normalization_7_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes~
|2z	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : : : @:@:@:@:@:@:@@:@:@:@:@:@:@?:?:?:?:?:?:?@:@:@:@:@:@:@@:@:@:@:@:@:@ : : : : : : :::::: : : : : : : : : : : : : : @:@:@:@:@@:@:@:@:@?:?:?:?:?@:@:@:@:@@:@:@:@:@ : : : : :::: : : : : @:@:@:@:@@:@:@:@:@?:?:?:?:?@:@:@:@:@@:@:@:@:@ : : : : :::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:-)
'
_output_shapes
:?@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@:  

_output_shapes
:@: !

_output_shapes
:@: "

_output_shapes
:@: #

_output_shapes
:@: $

_output_shapes
:@:,%(
&
_output_shapes
:@ : &

_output_shapes
: : '

_output_shapes
: : (

_output_shapes
: : )

_output_shapes
: : *

_output_shapes
: :,+(
&
_output_shapes
: : ,

_output_shapes
:: -

_output_shapes
:: .

_output_shapes
:: /

_output_shapes
:: 0

_output_shapes
::1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: :,:(
&
_output_shapes
: : ;

_output_shapes
: : <

_output_shapes
: : =

_output_shapes
: :,>(
&
_output_shapes
: @: ?

_output_shapes
:@: @

_output_shapes
:@: A

_output_shapes
:@:,B(
&
_output_shapes
:@@: C

_output_shapes
:@: D

_output_shapes
:@: E

_output_shapes
:@:-F)
'
_output_shapes
:@?:!G

_output_shapes	
:?:!H

_output_shapes	
:?:!I

_output_shapes	
:?:-J)
'
_output_shapes
:?@: K

_output_shapes
:@: L

_output_shapes
:@: M

_output_shapes
:@:,N(
&
_output_shapes
:@@: O

_output_shapes
:@: P

_output_shapes
:@: Q

_output_shapes
:@:,R(
&
_output_shapes
:@ : S

_output_shapes
: : T

_output_shapes
: : U

_output_shapes
: :,V(
&
_output_shapes
: : W

_output_shapes
:: X

_output_shapes
:: Y

_output_shapes
::,Z(
&
_output_shapes
: : [

_output_shapes
: : \

_output_shapes
: : ]

_output_shapes
: :,^(
&
_output_shapes
: @: _

_output_shapes
:@: `

_output_shapes
:@: a

_output_shapes
:@:,b(
&
_output_shapes
:@@: c

_output_shapes
:@: d

_output_shapes
:@: e

_output_shapes
:@:-f)
'
_output_shapes
:@?:!g

_output_shapes	
:?:!h

_output_shapes	
:?:!i

_output_shapes	
:?:-j)
'
_output_shapes
:?@: k

_output_shapes
:@: l

_output_shapes
:@: m

_output_shapes
:@:,n(
&
_output_shapes
:@@: o

_output_shapes
:@: p

_output_shapes
:@: q

_output_shapes
:@:,r(
&
_output_shapes
:@ : s

_output_shapes
: : t

_output_shapes
: : u

_output_shapes
: :,v(
&
_output_shapes
: : w

_output_shapes
:: x

_output_shapes
:: y

_output_shapes
::z

_output_shapes
: 
?
H
,__inference_activation_7_layer_call_fn_10941

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_7_layer_call_and_return_conditional_losses_69932
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
B__inference_conv2d_5_layer_call_and_return_conditional_losses_6763

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_1_layer_call_fn_9723

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_75912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_8134
input_1!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?%

unknown_23:?@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:@

unknown_28:@$

unknown_29:@@

unknown_30:@

unknown_31:@

unknown_32:@

unknown_33:@

unknown_34:@$

unknown_35:@ 

unknown_36: 

unknown_37: 

unknown_38: 

unknown_39: 

unknown_40: $

unknown_41: 

unknown_42:

unknown_43:

unknown_44:

unknown_45:

unknown_46:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*B
_read_only_resource_inputs$
" 	
 !"%&'(+,-.*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_79342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes
}:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6690

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  @2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_6400

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:??????????? 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
5__inference_batch_normalization_3_layer_call_fn_10089

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_74252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????  ?: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_9434

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9836

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
V__inference_max_pooling_with_argmax2d_3_layer_call_and_return_conditional_losses_10107

inputs
identity

identity_1?
MaxPoolWithArgmaxMaxPoolWithArgmaxinputs*
T0*L
_output_shapes:
8:??????????:??????????*
ksize
*
paddingSAME*
strides
2
MaxPoolWithArgmaxz
CastCastMaxPoolWithArgmax:argmax:0*

DstT0*

SrcT0	*0
_output_shapes
:??????????2
Castw
IdentityIdentityMaxPoolWithArgmax:output:0*
T0*0
_output_shapes
:??????????2

Identityi

Identity_1IdentityCast:y:0*
T0*0
_output_shapes
:??????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_7508

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?	
?
2__inference_batch_normalization_layer_call_fn_9501

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_53742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
"__inference_signature_wrapper_8515
input_1!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?%

unknown_23:?@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:@

unknown_28:@$

unknown_29:@@

unknown_30:@

unknown_31:@

unknown_32:@

unknown_33:@

unknown_34:@$

unknown_35:@ 

unknown_36: 

unknown_37: 

unknown_38: 

unknown_39: 

unknown_40: $

unknown_41: 

unknown_42:

unknown_43:

unknown_44:

unknown_45:

unknown_46:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_53522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes
}:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
`
D__inference_activation_layer_call_and_return_conditional_losses_9545

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:??????????? 2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
??
?
?__inference_model_layer_call_and_return_conditional_losses_6996

inputs%
conv2d_6378: 
conv2d_6380: &
batch_normalization_6401: &
batch_normalization_6403: &
batch_normalization_6405: &
batch_normalization_6407: '
conv2d_1_6439: @
conv2d_1_6441:@(
batch_normalization_1_6462:@(
batch_normalization_1_6464:@(
batch_normalization_1_6466:@(
batch_normalization_1_6468:@'
conv2d_2_6500:@@
conv2d_2_6502:@(
batch_normalization_2_6523:@(
batch_normalization_2_6525:@(
batch_normalization_2_6527:@(
batch_normalization_2_6529:@(
conv2d_3_6561:@?
conv2d_3_6563:	?)
batch_normalization_3_6584:	?)
batch_normalization_3_6586:	?)
batch_normalization_3_6588:	?)
batch_normalization_3_6590:	?(
conv2d_4_6668:?@
conv2d_4_6670:@(
batch_normalization_4_6691:@(
batch_normalization_4_6693:@(
batch_normalization_4_6695:@(
batch_normalization_4_6697:@'
conv2d_5_6764:@@
conv2d_5_6766:@(
batch_normalization_5_6787:@(
batch_normalization_5_6789:@(
batch_normalization_5_6791:@(
batch_normalization_5_6793:@'
conv2d_6_6860:@ 
conv2d_6_6862: (
batch_normalization_6_6883: (
batch_normalization_6_6885: (
batch_normalization_6_6887: (
batch_normalization_6_6889: '
conv2d_7_6956: 
conv2d_7_6958:(
batch_normalization_7_6979:(
batch_normalization_7_6981:(
batch_normalization_7_6983:(
batch_normalization_7_6985:
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?-batch_normalization_5/StatefulPartitionedCall?-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_6378conv2d_6380*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_63772 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_6401batch_normalization_6403batch_normalization_6405batch_normalization_6407*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_64002-
+batch_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_64152
activation/PartitionedCall?
)max_pooling_with_argmax2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::??????????? :??????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_max_pooling_with_argmax2d_layer_call_and_return_conditional_losses_64252+
)max_pooling_with_argmax2d/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall2max_pooling_with_argmax2d/PartitionedCall:output:0conv2d_1_6439conv2d_1_6441*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_64382"
 conv2d_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_6462batch_normalization_1_6464batch_normalization_1_6466batch_normalization_1_6468*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_64612/
-batch_normalization_1/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_64762
activation_1/PartitionedCall?
+max_pooling_with_argmax2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:?????????@@@:?????????@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_max_pooling_with_argmax2d_1_layer_call_and_return_conditional_losses_64862-
+max_pooling_with_argmax2d_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall4max_pooling_with_argmax2d_1/PartitionedCall:output:0conv2d_2_6500conv2d_2_6502*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_64992"
 conv2d_2/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_6523batch_normalization_2_6525batch_normalization_2_6527batch_normalization_2_6529*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_65222/
-batch_normalization_2/StatefulPartitionedCall?
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_65372
activation_2/PartitionedCall?
+max_pooling_with_argmax2d_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:?????????  @:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_max_pooling_with_argmax2d_2_layer_call_and_return_conditional_losses_65472-
+max_pooling_with_argmax2d_2/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall4max_pooling_with_argmax2d_2/PartitionedCall:output:0conv2d_3_6561conv2d_3_6563*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_65602"
 conv2d_3/StatefulPartitionedCall?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_6584batch_normalization_3_6586batch_normalization_3_6588batch_normalization_3_6590*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_65832/
-batch_normalization_3/StatefulPartitionedCall?
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_65982
activation_3/PartitionedCall?
+max_pooling_with_argmax2d_3/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *L
_output_shapes:
8:??????????:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_max_pooling_with_argmax2d_3_layer_call_and_return_conditional_losses_66082-
+max_pooling_with_argmax2d_3/PartitionedCall?
max_unpooling2d/PartitionedCallPartitionedCall4max_pooling_with_argmax2d_3/PartitionedCall:output:04max_pooling_with_argmax2d_3/PartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_unpooling2d_layer_call_and_return_conditional_losses_66552!
max_unpooling2d/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_unpooling2d/PartitionedCall:output:0conv2d_4_6668conv2d_4_6670*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_66672"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_6691batch_normalization_4_6693batch_normalization_4_6695batch_normalization_4_6697*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_66902/
-batch_normalization_4/StatefulPartitionedCall?
activation_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_67052
activation_4/PartitionedCall?
!max_unpooling2d_1/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:04max_pooling_with_argmax2d_2/PartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_unpooling2d_1_layer_call_and_return_conditional_losses_67512#
!max_unpooling2d_1/PartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall*max_unpooling2d_1/PartitionedCall:output:0conv2d_5_6764conv2d_5_6766*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_67632"
 conv2d_5/StatefulPartitionedCall?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_6787batch_normalization_5_6789batch_normalization_5_6791batch_normalization_5_6793*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_67862/
-batch_normalization_5/StatefulPartitionedCall?
activation_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_5_layer_call_and_return_conditional_losses_68012
activation_5/PartitionedCall?
!max_unpooling2d_2/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:04max_pooling_with_argmax2d_1/PartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_unpooling2d_2_layer_call_and_return_conditional_losses_68472#
!max_unpooling2d_2/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall*max_unpooling2d_2/PartitionedCall:output:0conv2d_6_6860conv2d_6_6862*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_68592"
 conv2d_6/StatefulPartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_6883batch_normalization_6_6885batch_normalization_6_6887batch_normalization_6_6889*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_68822/
-batch_normalization_6/StatefulPartitionedCall?
activation_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_6_layer_call_and_return_conditional_losses_68972
activation_6/PartitionedCall?
!max_unpooling2d_3/PartitionedCallPartitionedCall%activation_6/PartitionedCall:output:02max_pooling_with_argmax2d/PartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_unpooling2d_3_layer_call_and_return_conditional_losses_69432#
!max_unpooling2d_3/PartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall*max_unpooling2d_3/PartitionedCall:output:0conv2d_7_6956conv2d_7_6958*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_7_layer_call_and_return_conditional_losses_69552"
 conv2d_7/StatefulPartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_7_6979batch_normalization_7_6981batch_normalization_7_6983batch_normalization_7_6985*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_69782/
-batch_normalization_7/StatefulPartitionedCall?
activation_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_7_layer_call_and_return_conditional_losses_69932
activation_7/PartitionedCall?
IdentityIdentity%activation_7/PartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes
}:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_9296

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?%

unknown_23:?@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:@

unknown_28:@$

unknown_29:@@

unknown_30:@

unknown_31:@

unknown_32:@

unknown_33:@

unknown_34:@$

unknown_35:@ 

unknown_36: 

unknown_37: 

unknown_38: 

unknown_39: 

unknown_40: $

unknown_41: 

unknown_42:

unknown_43:

unknown_44:

unknown_45:

unknown_46:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_69962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes
}:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
F__inference_activation_7_layer_call_and_return_conditional_losses_6993

inputs
identitya
SigmoidSigmoidinputs*
T0*1
_output_shapes
:???????????2	
Sigmoidi
IdentityIdentitySigmoid:y:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
B__inference_conv2d_2_layer_call_and_return_conditional_losses_9773

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
]
1__inference_max_unpooling2d_1_layer_call_fn_10382
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_unpooling2d_1_layer_call_and_return_conditional_losses_67512
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????  @:?????????  @:Y U
/
_output_shapes
:?????????  @
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????  @
"
_user_specified_name
inputs/1
?
?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_9983

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
H
,__inference_activation_4_layer_call_fn_10332

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_67052
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????  @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
5__inference_batch_normalization_5_layer_call_fn_10525

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_72682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
]
1__inference_max_unpooling2d_2_layer_call_fn_10585
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_unpooling2d_2_layer_call_and_return_conditional_losses_68472
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????@@@:?????????@@@:Y U
/
_output_shapes
:?????????@@@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????@@@
"
_user_specified_name
inputs/1
?
?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5626

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
B__inference_conv2d_1_layer_call_and_return_conditional_losses_9590

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
'__inference_conv2d_3_layer_call_fn_9965

inputs"
unknown:@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_65602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?	
?
4__inference_batch_normalization_1_layer_call_fn_9697

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_55442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_7591

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5500

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_1:
serving_default_input_1:0???????????J
activation_7:
StatefulPartitionedCall:0???????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer-15
layer-16
layer-17
layer_with_weights-8
layer-18
layer_with_weights-9
layer-19
layer-20
layer-21
layer_with_weights-10
layer-22
layer_with_weights-11
layer-23
layer-24
layer-25
layer_with_weights-12
layer-26
layer_with_weights-13
layer-27
layer-28
layer-29
layer_with_weights-14
layer-30
 layer_with_weights-15
 layer-31
!layer-32
"	optimizer
#regularization_losses
$trainable_variables
%	variables
&	keras_api
'
signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
?

(kernel
)bias
*regularization_losses
+trainable_variables
,	variables
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
.axis
	/gamma
0beta
1moving_mean
2moving_variance
3regularization_losses
4trainable_variables
5	variables
6	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
7regularization_losses
8trainable_variables
9	variables
:	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
;regularization_losses
<trainable_variables
=	variables
>	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

?kernel
@bias
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
Eaxis
	Fgamma
Gbeta
Hmoving_mean
Imoving_variance
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

Vkernel
Wbias
Xregularization_losses
Ytrainable_variables
Z	variables
[	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
\axis
	]gamma
^beta
_moving_mean
`moving_variance
aregularization_losses
btrainable_variables
c	variables
d	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
eregularization_losses
ftrainable_variables
g	variables
h	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
iregularization_losses
jtrainable_variables
k	variables
l	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

mkernel
nbias
oregularization_losses
ptrainable_variables
q	variables
r	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
saxis
	tgamma
ubeta
vmoving_mean
wmoving_variance
xregularization_losses
ytrainable_variables
z	variables
{	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
|regularization_losses
}trainable_variables
~	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate(m?)m?/m?0m??m?@m?Fm?Gm?Vm?Wm?]m?^m?mm?nm?tm?um?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?(v?)v?/v?0v??v?@v?Fv?Gv?Vv?Wv?]v?^v?mv?nv?tv?uv?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
 "
trackable_list_wrapper
?
(0
)1
/2
03
?4
@5
F6
G7
V8
W9
]10
^11
m12
n13
t14
u15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31"
trackable_list_wrapper
?
(0
)1
/2
03
14
25
?6
@7
F8
G9
H10
I11
V12
W13
]14
^15
_16
`17
m18
n19
t20
u21
v22
w23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47"
trackable_list_wrapper
?
#regularization_losses
?layer_metrics
$trainable_variables
 ?layer_regularization_losses
%	variables
?metrics
?non_trainable_variables
?layers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
':% 2conv2d/kernel
: 2conv2d/bias
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
?
*regularization_losses
?layer_metrics
+trainable_variables
 ?layer_regularization_losses
,	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':% 2batch_normalization/gamma
&:$ 2batch_normalization/beta
/:-  (2batch_normalization/moving_mean
3:1  (2#batch_normalization/moving_variance
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
<
/0
01
12
23"
trackable_list_wrapper
?
3regularization_losses
?layer_metrics
4trainable_variables
 ?layer_regularization_losses
5	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
7regularization_losses
?layer_metrics
8trainable_variables
 ?layer_regularization_losses
9	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
;regularization_losses
?layer_metrics
<trainable_variables
 ?layer_regularization_losses
=	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):' @2conv2d_1/kernel
:@2conv2d_1/bias
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
?
Aregularization_losses
?layer_metrics
Btrainable_variables
 ?layer_regularization_losses
C	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_1/gamma
(:&@2batch_normalization_1/beta
1:/@ (2!batch_normalization_1/moving_mean
5:3@ (2%batch_normalization_1/moving_variance
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
<
F0
G1
H2
I3"
trackable_list_wrapper
?
Jregularization_losses
?layer_metrics
Ktrainable_variables
 ?layer_regularization_losses
L	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Nregularization_losses
?layer_metrics
Otrainable_variables
 ?layer_regularization_losses
P	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Rregularization_losses
?layer_metrics
Strainable_variables
 ?layer_regularization_losses
T	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'@@2conv2d_2/kernel
:@2conv2d_2/bias
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
?
Xregularization_losses
?layer_metrics
Ytrainable_variables
 ?layer_regularization_losses
Z	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_2/gamma
(:&@2batch_normalization_2/beta
1:/@ (2!batch_normalization_2/moving_mean
5:3@ (2%batch_normalization_2/moving_variance
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
<
]0
^1
_2
`3"
trackable_list_wrapper
?
aregularization_losses
?layer_metrics
btrainable_variables
 ?layer_regularization_losses
c	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
eregularization_losses
?layer_metrics
ftrainable_variables
 ?layer_regularization_losses
g	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
iregularization_losses
?layer_metrics
jtrainable_variables
 ?layer_regularization_losses
k	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(@?2conv2d_3/kernel
:?2conv2d_3/bias
 "
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
?
oregularization_losses
?layer_metrics
ptrainable_variables
 ?layer_regularization_losses
q	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(?2batch_normalization_3/gamma
):'?2batch_normalization_3/beta
2:0? (2!batch_normalization_3/moving_mean
6:4? (2%batch_normalization_3/moving_variance
 "
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
<
t0
u1
v2
w3"
trackable_list_wrapper
?
xregularization_losses
?layer_metrics
ytrainable_variables
 ?layer_regularization_losses
z	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
|regularization_losses
?layer_metrics
}trainable_variables
 ?layer_regularization_losses
~	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(?@2conv2d_4/kernel
:@2conv2d_4/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_4/gamma
(:&@2batch_normalization_4/beta
1:/@ (2!batch_normalization_4/moving_mean
5:3@ (2%batch_normalization_4/moving_variance
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'@@2conv2d_5/kernel
:@2conv2d_5/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_5/gamma
(:&@2batch_normalization_5/beta
1:/@ (2!batch_normalization_5/moving_mean
5:3@ (2%batch_normalization_5/moving_variance
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'@ 2conv2d_6/kernel
: 2conv2d_6/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_6/gamma
(:& 2batch_normalization_6/beta
1:/  (2!batch_normalization_6/moving_mean
5:3  (2%batch_normalization_6/moving_variance
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):' 2conv2d_7/kernel
:2conv2d_7/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_7/gamma
(:&2batch_normalization_7/beta
1:/ (2!batch_normalization_7/moving_mean
5:3 (2%batch_normalization_7/moving_variance
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?	variables
?metrics
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
10
21
H2
I3
_4
`5
v6
w7
?8
?9
?10
?11
?12
?13
?14
?15"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32"
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
.
10
21"
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
.
H0
I1"
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
.
_0
`1"
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
.
v0
w1"
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
0
?0
?1"
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
0
?0
?1"
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
0
?0
?1"
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
0
?0
?1"
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
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
,:* 2Adam/conv2d/kernel/m
: 2Adam/conv2d/bias/m
,:* 2 Adam/batch_normalization/gamma/m
+:) 2Adam/batch_normalization/beta/m
.:, @2Adam/conv2d_1/kernel/m
 :@2Adam/conv2d_1/bias/m
.:,@2"Adam/batch_normalization_1/gamma/m
-:+@2!Adam/batch_normalization_1/beta/m
.:,@@2Adam/conv2d_2/kernel/m
 :@2Adam/conv2d_2/bias/m
.:,@2"Adam/batch_normalization_2/gamma/m
-:+@2!Adam/batch_normalization_2/beta/m
/:-@?2Adam/conv2d_3/kernel/m
!:?2Adam/conv2d_3/bias/m
/:-?2"Adam/batch_normalization_3/gamma/m
.:,?2!Adam/batch_normalization_3/beta/m
/:-?@2Adam/conv2d_4/kernel/m
 :@2Adam/conv2d_4/bias/m
.:,@2"Adam/batch_normalization_4/gamma/m
-:+@2!Adam/batch_normalization_4/beta/m
.:,@@2Adam/conv2d_5/kernel/m
 :@2Adam/conv2d_5/bias/m
.:,@2"Adam/batch_normalization_5/gamma/m
-:+@2!Adam/batch_normalization_5/beta/m
.:,@ 2Adam/conv2d_6/kernel/m
 : 2Adam/conv2d_6/bias/m
.:, 2"Adam/batch_normalization_6/gamma/m
-:+ 2!Adam/batch_normalization_6/beta/m
.:, 2Adam/conv2d_7/kernel/m
 :2Adam/conv2d_7/bias/m
.:,2"Adam/batch_normalization_7/gamma/m
-:+2!Adam/batch_normalization_7/beta/m
,:* 2Adam/conv2d/kernel/v
: 2Adam/conv2d/bias/v
,:* 2 Adam/batch_normalization/gamma/v
+:) 2Adam/batch_normalization/beta/v
.:, @2Adam/conv2d_1/kernel/v
 :@2Adam/conv2d_1/bias/v
.:,@2"Adam/batch_normalization_1/gamma/v
-:+@2!Adam/batch_normalization_1/beta/v
.:,@@2Adam/conv2d_2/kernel/v
 :@2Adam/conv2d_2/bias/v
.:,@2"Adam/batch_normalization_2/gamma/v
-:+@2!Adam/batch_normalization_2/beta/v
/:-@?2Adam/conv2d_3/kernel/v
!:?2Adam/conv2d_3/bias/v
/:-?2"Adam/batch_normalization_3/gamma/v
.:,?2!Adam/batch_normalization_3/beta/v
/:-?@2Adam/conv2d_4/kernel/v
 :@2Adam/conv2d_4/bias/v
.:,@2"Adam/batch_normalization_4/gamma/v
-:+@2!Adam/batch_normalization_4/beta/v
.:,@@2Adam/conv2d_5/kernel/v
 :@2Adam/conv2d_5/bias/v
.:,@2"Adam/batch_normalization_5/gamma/v
-:+@2!Adam/batch_normalization_5/beta/v
.:,@ 2Adam/conv2d_6/kernel/v
 : 2Adam/conv2d_6/bias/v
.:, 2"Adam/batch_normalization_6/gamma/v
-:+ 2!Adam/batch_normalization_6/beta/v
.:, 2Adam/conv2d_7/kernel/v
 :2Adam/conv2d_7/bias/v
.:,2"Adam/batch_normalization_7/gamma/v
-:+2!Adam/batch_normalization_7/beta/v
?2?
?__inference_model_layer_call_and_return_conditional_losses_8855
?__inference_model_layer_call_and_return_conditional_losses_9195
?__inference_model_layer_call_and_return_conditional_losses_8270
?__inference_model_layer_call_and_return_conditional_losses_8406?
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

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
$__inference_model_layer_call_fn_7095
$__inference_model_layer_call_fn_9296
$__inference_model_layer_call_fn_9397
$__inference_model_layer_call_fn_8134?
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

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
__inference__wrapped_model_5352input_1"?
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
 
?2?
@__inference_conv2d_layer_call_and_return_conditional_losses_9407?
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
?2?
%__inference_conv2d_layer_call_fn_9416?
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
?2?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_9434
M__inference_batch_normalization_layer_call_and_return_conditional_losses_9452
M__inference_batch_normalization_layer_call_and_return_conditional_losses_9470
M__inference_batch_normalization_layer_call_and_return_conditional_losses_9488?
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

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
2__inference_batch_normalization_layer_call_fn_9501
2__inference_batch_normalization_layer_call_fn_9514
2__inference_batch_normalization_layer_call_fn_9527
2__inference_batch_normalization_layer_call_fn_9540?
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

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_activation_layer_call_and_return_conditional_losses_9545?
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
?2?
)__inference_activation_layer_call_fn_9550?
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
?2?
S__inference_max_pooling_with_argmax2d_layer_call_and_return_conditional_losses_9558
S__inference_max_pooling_with_argmax2d_layer_call_and_return_conditional_losses_9566?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
8__inference_max_pooling_with_argmax2d_layer_call_fn_9573
8__inference_max_pooling_with_argmax2d_layer_call_fn_9580?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
B__inference_conv2d_1_layer_call_and_return_conditional_losses_9590?
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
?2?
'__inference_conv2d_1_layer_call_fn_9599?
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
?2?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9617
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9635
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9653
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9671?
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

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
4__inference_batch_normalization_1_layer_call_fn_9684
4__inference_batch_normalization_1_layer_call_fn_9697
4__inference_batch_normalization_1_layer_call_fn_9710
4__inference_batch_normalization_1_layer_call_fn_9723?
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

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_activation_1_layer_call_and_return_conditional_losses_9728?
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
?2?
+__inference_activation_1_layer_call_fn_9733?
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
?2?
U__inference_max_pooling_with_argmax2d_1_layer_call_and_return_conditional_losses_9741
U__inference_max_pooling_with_argmax2d_1_layer_call_and_return_conditional_losses_9749?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
:__inference_max_pooling_with_argmax2d_1_layer_call_fn_9756
:__inference_max_pooling_with_argmax2d_1_layer_call_fn_9763?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
B__inference_conv2d_2_layer_call_and_return_conditional_losses_9773?
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
?2?
'__inference_conv2d_2_layer_call_fn_9782?
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
?2?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9800
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9818
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9836
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9854?
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

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
4__inference_batch_normalization_2_layer_call_fn_9867
4__inference_batch_normalization_2_layer_call_fn_9880
4__inference_batch_normalization_2_layer_call_fn_9893
4__inference_batch_normalization_2_layer_call_fn_9906?
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

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_activation_2_layer_call_and_return_conditional_losses_9911?
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
?2?
+__inference_activation_2_layer_call_fn_9916?
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
?2?
U__inference_max_pooling_with_argmax2d_2_layer_call_and_return_conditional_losses_9924
U__inference_max_pooling_with_argmax2d_2_layer_call_and_return_conditional_losses_9932?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
:__inference_max_pooling_with_argmax2d_2_layer_call_fn_9939
:__inference_max_pooling_with_argmax2d_2_layer_call_fn_9946?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
B__inference_conv2d_3_layer_call_and_return_conditional_losses_9956?
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
?2?
'__inference_conv2d_3_layer_call_fn_9965?
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
?2?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_9983
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_10001
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_10019
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_10037?
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

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
5__inference_batch_normalization_3_layer_call_fn_10050
5__inference_batch_normalization_3_layer_call_fn_10063
5__inference_batch_normalization_3_layer_call_fn_10076
5__inference_batch_normalization_3_layer_call_fn_10089?
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

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_activation_3_layer_call_and_return_conditional_losses_10094?
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
?2?
,__inference_activation_3_layer_call_fn_10099?
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
?2?
V__inference_max_pooling_with_argmax2d_3_layer_call_and_return_conditional_losses_10107
V__inference_max_pooling_with_argmax2d_3_layer_call_and_return_conditional_losses_10115?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
;__inference_max_pooling_with_argmax2d_3_layer_call_fn_10122
;__inference_max_pooling_with_argmax2d_3_layer_call_fn_10129?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
J__inference_max_unpooling2d_layer_call_and_return_conditional_losses_10173?
???
FullArgSpec-
args%?"
jself
jinputs
joutput_shape
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_max_unpooling2d_layer_call_fn_10179?
???
FullArgSpec-
args%?"
jself
jinputs
joutput_shape
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_10189?
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
?2?
(__inference_conv2d_4_layer_call_fn_10198?
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
?2?
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_10216
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_10234
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_10252
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_10270?
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

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
5__inference_batch_normalization_4_layer_call_fn_10283
5__inference_batch_normalization_4_layer_call_fn_10296
5__inference_batch_normalization_4_layer_call_fn_10309
5__inference_batch_normalization_4_layer_call_fn_10322?
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

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_activation_4_layer_call_and_return_conditional_losses_10327?
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
?2?
,__inference_activation_4_layer_call_fn_10332?
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
?2?
L__inference_max_unpooling2d_1_layer_call_and_return_conditional_losses_10376?
???
FullArgSpec-
args%?"
jself
jinputs
joutput_shape
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_max_unpooling2d_1_layer_call_fn_10382?
???
FullArgSpec-
args%?"
jself
jinputs
joutput_shape
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_5_layer_call_and_return_conditional_losses_10392?
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
?2?
(__inference_conv2d_5_layer_call_fn_10401?
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
?2?
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_10419
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_10437
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_10455
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_10473?
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

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
5__inference_batch_normalization_5_layer_call_fn_10486
5__inference_batch_normalization_5_layer_call_fn_10499
5__inference_batch_normalization_5_layer_call_fn_10512
5__inference_batch_normalization_5_layer_call_fn_10525?
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

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_activation_5_layer_call_and_return_conditional_losses_10530?
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
?2?
,__inference_activation_5_layer_call_fn_10535?
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
?2?
L__inference_max_unpooling2d_2_layer_call_and_return_conditional_losses_10579?
???
FullArgSpec-
args%?"
jself
jinputs
joutput_shape
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_max_unpooling2d_2_layer_call_fn_10585?
???
FullArgSpec-
args%?"
jself
jinputs
joutput_shape
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_6_layer_call_and_return_conditional_losses_10595?
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
?2?
(__inference_conv2d_6_layer_call_fn_10604?
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
?2?
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10622
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10640
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10658
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10676?
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

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
5__inference_batch_normalization_6_layer_call_fn_10689
5__inference_batch_normalization_6_layer_call_fn_10702
5__inference_batch_normalization_6_layer_call_fn_10715
5__inference_batch_normalization_6_layer_call_fn_10728?
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

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_activation_6_layer_call_and_return_conditional_losses_10733?
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
?2?
,__inference_activation_6_layer_call_fn_10738?
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
?2?
L__inference_max_unpooling2d_3_layer_call_and_return_conditional_losses_10782?
???
FullArgSpec-
args%?"
jself
jinputs
joutput_shape
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_max_unpooling2d_3_layer_call_fn_10788?
???
FullArgSpec-
args%?"
jself
jinputs
joutput_shape
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_7_layer_call_and_return_conditional_losses_10798?
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
?2?
(__inference_conv2d_7_layer_call_fn_10807?
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
?2?
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10825
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10843
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10861
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10879?
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

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
5__inference_batch_normalization_7_layer_call_fn_10892
5__inference_batch_normalization_7_layer_call_fn_10905
5__inference_batch_normalization_7_layer_call_fn_10918
5__inference_batch_normalization_7_layer_call_fn_10931?
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

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_activation_7_layer_call_and_return_conditional_losses_10936?
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
?2?
,__inference_activation_7_layer_call_fn_10941?
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
?B?
"__inference_signature_wrapper_8515input_1"?
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
 ?
__inference__wrapped_model_5352?H()/012?@FGHIVW]^_`mntuvw????????????????????????:?7
0?-
+?(
input_1???????????
? "E?B
@
activation_70?-
activation_7????????????
F__inference_activation_1_layer_call_and_return_conditional_losses_9728l9?6
/?,
*?'
inputs???????????@
? "/?,
%?"
0???????????@
? ?
+__inference_activation_1_layer_call_fn_9733_9?6
/?,
*?'
inputs???????????@
? ""????????????@?
F__inference_activation_2_layer_call_and_return_conditional_losses_9911h7?4
-?*
(?%
inputs?????????@@@
? "-?*
#? 
0?????????@@@
? ?
+__inference_activation_2_layer_call_fn_9916[7?4
-?*
(?%
inputs?????????@@@
? " ??????????@@@?
G__inference_activation_3_layer_call_and_return_conditional_losses_10094j8?5
.?+
)?&
inputs?????????  ?
? ".?+
$?!
0?????????  ?
? ?
,__inference_activation_3_layer_call_fn_10099]8?5
.?+
)?&
inputs?????????  ?
? "!??????????  ??
G__inference_activation_4_layer_call_and_return_conditional_losses_10327h7?4
-?*
(?%
inputs?????????  @
? "-?*
#? 
0?????????  @
? ?
,__inference_activation_4_layer_call_fn_10332[7?4
-?*
(?%
inputs?????????  @
? " ??????????  @?
G__inference_activation_5_layer_call_and_return_conditional_losses_10530h7?4
-?*
(?%
inputs?????????@@@
? "-?*
#? 
0?????????@@@
? ?
,__inference_activation_5_layer_call_fn_10535[7?4
-?*
(?%
inputs?????????@@@
? " ??????????@@@?
G__inference_activation_6_layer_call_and_return_conditional_losses_10733l9?6
/?,
*?'
inputs??????????? 
? "/?,
%?"
0??????????? 
? ?
,__inference_activation_6_layer_call_fn_10738_9?6
/?,
*?'
inputs??????????? 
? ""???????????? ?
G__inference_activation_7_layer_call_and_return_conditional_losses_10936l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
,__inference_activation_7_layer_call_fn_10941_9?6
/?,
*?'
inputs???????????
? ""?????????????
D__inference_activation_layer_call_and_return_conditional_losses_9545l9?6
/?,
*?'
inputs??????????? 
? "/?,
%?"
0??????????? 
? ?
)__inference_activation_layer_call_fn_9550_9?6
/?,
*?'
inputs??????????? 
? ""???????????? ?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9617?FGHIM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9635?FGHIM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9653vFGHI=?:
3?0
*?'
inputs???????????@
p 
? "/?,
%?"
0???????????@
? ?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9671vFGHI=?:
3?0
*?'
inputs???????????@
p
? "/?,
%?"
0???????????@
? ?
4__inference_batch_normalization_1_layer_call_fn_9684?FGHIM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
4__inference_batch_normalization_1_layer_call_fn_9697?FGHIM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
4__inference_batch_normalization_1_layer_call_fn_9710iFGHI=?:
3?0
*?'
inputs???????????@
p 
? ""????????????@?
4__inference_batch_normalization_1_layer_call_fn_9723iFGHI=?:
3?0
*?'
inputs???????????@
p
? ""????????????@?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9800?]^_`M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9818?]^_`M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9836r]^_`;?8
1?.
(?%
inputs?????????@@@
p 
? "-?*
#? 
0?????????@@@
? ?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9854r]^_`;?8
1?.
(?%
inputs?????????@@@
p
? "-?*
#? 
0?????????@@@
? ?
4__inference_batch_normalization_2_layer_call_fn_9867?]^_`M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
4__inference_batch_normalization_2_layer_call_fn_9880?]^_`M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
4__inference_batch_normalization_2_layer_call_fn_9893e]^_`;?8
1?.
(?%
inputs?????????@@@
p 
? " ??????????@@@?
4__inference_batch_normalization_2_layer_call_fn_9906e]^_`;?8
1?.
(?%
inputs?????????@@@
p
? " ??????????@@@?
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_10001?tuvwN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_10019ttuvw<?9
2?/
)?&
inputs?????????  ?
p 
? ".?+
$?!
0?????????  ?
? ?
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_10037ttuvw<?9
2?/
)?&
inputs?????????  ?
p
? ".?+
$?!
0?????????  ?
? ?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_9983?tuvwN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
5__inference_batch_normalization_3_layer_call_fn_10050?tuvwN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
5__inference_batch_normalization_3_layer_call_fn_10063?tuvwN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
5__inference_batch_normalization_3_layer_call_fn_10076gtuvw<?9
2?/
)?&
inputs?????????  ?
p 
? "!??????????  ??
5__inference_batch_normalization_3_layer_call_fn_10089gtuvw<?9
2?/
)?&
inputs?????????  ?
p
? "!??????????  ??
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_10216?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_10234?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_10252v????;?8
1?.
(?%
inputs?????????  @
p 
? "-?*
#? 
0?????????  @
? ?
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_10270v????;?8
1?.
(?%
inputs?????????  @
p
? "-?*
#? 
0?????????  @
? ?
5__inference_batch_normalization_4_layer_call_fn_10283?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
5__inference_batch_normalization_4_layer_call_fn_10296?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
5__inference_batch_normalization_4_layer_call_fn_10309i????;?8
1?.
(?%
inputs?????????  @
p 
? " ??????????  @?
5__inference_batch_normalization_4_layer_call_fn_10322i????;?8
1?.
(?%
inputs?????????  @
p
? " ??????????  @?
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_10419?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_10437?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_10455v????;?8
1?.
(?%
inputs?????????@@@
p 
? "-?*
#? 
0?????????@@@
? ?
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_10473v????;?8
1?.
(?%
inputs?????????@@@
p
? "-?*
#? 
0?????????@@@
? ?
5__inference_batch_normalization_5_layer_call_fn_10486?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
5__inference_batch_normalization_5_layer_call_fn_10499?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
5__inference_batch_normalization_5_layer_call_fn_10512i????;?8
1?.
(?%
inputs?????????@@@
p 
? " ??????????@@@?
5__inference_batch_normalization_5_layer_call_fn_10525i????;?8
1?.
(?%
inputs?????????@@@
p
? " ??????????@@@?
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10622?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10640?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10658z????=?:
3?0
*?'
inputs??????????? 
p 
? "/?,
%?"
0??????????? 
? ?
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_10676z????=?:
3?0
*?'
inputs??????????? 
p
? "/?,
%?"
0??????????? 
? ?
5__inference_batch_normalization_6_layer_call_fn_10689?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
5__inference_batch_normalization_6_layer_call_fn_10702?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
5__inference_batch_normalization_6_layer_call_fn_10715m????=?:
3?0
*?'
inputs??????????? 
p 
? ""???????????? ?
5__inference_batch_normalization_6_layer_call_fn_10728m????=?:
3?0
*?'
inputs??????????? 
p
? ""???????????? ?
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10825?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10843?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10861z????=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_10879z????=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
5__inference_batch_normalization_7_layer_call_fn_10892?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
5__inference_batch_normalization_7_layer_call_fn_10905?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
5__inference_batch_normalization_7_layer_call_fn_10918m????=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
5__inference_batch_normalization_7_layer_call_fn_10931m????=?:
3?0
*?'
inputs???????????
p
? ""?????????????
M__inference_batch_normalization_layer_call_and_return_conditional_losses_9434?/012M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_9452?/012M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_9470v/012=?:
3?0
*?'
inputs??????????? 
p 
? "/?,
%?"
0??????????? 
? ?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_9488v/012=?:
3?0
*?'
inputs??????????? 
p
? "/?,
%?"
0??????????? 
? ?
2__inference_batch_normalization_layer_call_fn_9501?/012M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
2__inference_batch_normalization_layer_call_fn_9514?/012M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
2__inference_batch_normalization_layer_call_fn_9527i/012=?:
3?0
*?'
inputs??????????? 
p 
? ""???????????? ?
2__inference_batch_normalization_layer_call_fn_9540i/012=?:
3?0
*?'
inputs??????????? 
p
? ""???????????? ?
B__inference_conv2d_1_layer_call_and_return_conditional_losses_9590p?@9?6
/?,
*?'
inputs??????????? 
? "/?,
%?"
0???????????@
? ?
'__inference_conv2d_1_layer_call_fn_9599c?@9?6
/?,
*?'
inputs??????????? 
? ""????????????@?
B__inference_conv2d_2_layer_call_and_return_conditional_losses_9773lVW7?4
-?*
(?%
inputs?????????@@@
? "-?*
#? 
0?????????@@@
? ?
'__inference_conv2d_2_layer_call_fn_9782_VW7?4
-?*
(?%
inputs?????????@@@
? " ??????????@@@?
B__inference_conv2d_3_layer_call_and_return_conditional_losses_9956mmn7?4
-?*
(?%
inputs?????????  @
? ".?+
$?!
0?????????  ?
? ?
'__inference_conv2d_3_layer_call_fn_9965`mn7?4
-?*
(?%
inputs?????????  @
? "!??????????  ??
C__inference_conv2d_4_layer_call_and_return_conditional_losses_10189o??8?5
.?+
)?&
inputs?????????  ?
? "-?*
#? 
0?????????  @
? ?
(__inference_conv2d_4_layer_call_fn_10198b??8?5
.?+
)?&
inputs?????????  ?
? " ??????????  @?
C__inference_conv2d_5_layer_call_and_return_conditional_losses_10392n??7?4
-?*
(?%
inputs?????????@@@
? "-?*
#? 
0?????????@@@
? ?
(__inference_conv2d_5_layer_call_fn_10401a??7?4
-?*
(?%
inputs?????????@@@
? " ??????????@@@?
C__inference_conv2d_6_layer_call_and_return_conditional_losses_10595r??9?6
/?,
*?'
inputs???????????@
? "/?,
%?"
0??????????? 
? ?
(__inference_conv2d_6_layer_call_fn_10604e??9?6
/?,
*?'
inputs???????????@
? ""???????????? ?
C__inference_conv2d_7_layer_call_and_return_conditional_losses_10798r??9?6
/?,
*?'
inputs??????????? 
? "/?,
%?"
0???????????
? ?
(__inference_conv2d_7_layer_call_fn_10807e??9?6
/?,
*?'
inputs??????????? 
? ""?????????????
@__inference_conv2d_layer_call_and_return_conditional_losses_9407p()9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0??????????? 
? ?
%__inference_conv2d_layer_call_fn_9416c()9?6
/?,
*?'
inputs???????????
? ""???????????? ?
U__inference_max_pooling_with_argmax2d_1_layer_call_and_return_conditional_losses_9741?I?F
/?,
*?'
inputs???????????@
?

trainingp "[?X
Q?N
%?"
0/0?????????@@@
%?"
0/1?????????@@@
? ?
U__inference_max_pooling_with_argmax2d_1_layer_call_and_return_conditional_losses_9749?I?F
/?,
*?'
inputs???????????@
?

trainingp"[?X
Q?N
%?"
0/0?????????@@@
%?"
0/1?????????@@@
? ?
:__inference_max_pooling_with_argmax2d_1_layer_call_fn_9756?I?F
/?,
*?'
inputs???????????@
?

trainingp "M?J
#? 
0?????????@@@
#? 
1?????????@@@?
:__inference_max_pooling_with_argmax2d_1_layer_call_fn_9763?I?F
/?,
*?'
inputs???????????@
?

trainingp"M?J
#? 
0?????????@@@
#? 
1?????????@@@?
U__inference_max_pooling_with_argmax2d_2_layer_call_and_return_conditional_losses_9924?G?D
-?*
(?%
inputs?????????@@@
?

trainingp "[?X
Q?N
%?"
0/0?????????  @
%?"
0/1?????????  @
? ?
U__inference_max_pooling_with_argmax2d_2_layer_call_and_return_conditional_losses_9932?G?D
-?*
(?%
inputs?????????@@@
?

trainingp"[?X
Q?N
%?"
0/0?????????  @
%?"
0/1?????????  @
? ?
:__inference_max_pooling_with_argmax2d_2_layer_call_fn_9939?G?D
-?*
(?%
inputs?????????@@@
?

trainingp "M?J
#? 
0?????????  @
#? 
1?????????  @?
:__inference_max_pooling_with_argmax2d_2_layer_call_fn_9946?G?D
-?*
(?%
inputs?????????@@@
?

trainingp"M?J
#? 
0?????????  @
#? 
1?????????  @?
V__inference_max_pooling_with_argmax2d_3_layer_call_and_return_conditional_losses_10107?H?E
.?+
)?&
inputs?????????  ?
?

trainingp "]?Z
S?P
&?#
0/0??????????
&?#
0/1??????????
? ?
V__inference_max_pooling_with_argmax2d_3_layer_call_and_return_conditional_losses_10115?H?E
.?+
)?&
inputs?????????  ?
?

trainingp"]?Z
S?P
&?#
0/0??????????
&?#
0/1??????????
? ?
;__inference_max_pooling_with_argmax2d_3_layer_call_fn_10122?H?E
.?+
)?&
inputs?????????  ?
?

trainingp "O?L
$?!
0??????????
$?!
1???????????
;__inference_max_pooling_with_argmax2d_3_layer_call_fn_10129?H?E
.?+
)?&
inputs?????????  ?
?

trainingp"O?L
$?!
0??????????
$?!
1???????????
S__inference_max_pooling_with_argmax2d_layer_call_and_return_conditional_losses_9558?I?F
/?,
*?'
inputs??????????? 
?

trainingp "_?\
U?R
'?$
0/0??????????? 
'?$
0/1??????????? 
? ?
S__inference_max_pooling_with_argmax2d_layer_call_and_return_conditional_losses_9566?I?F
/?,
*?'
inputs??????????? 
?

trainingp"_?\
U?R
'?$
0/0??????????? 
'?$
0/1??????????? 
? ?
8__inference_max_pooling_with_argmax2d_layer_call_fn_9573?I?F
/?,
*?'
inputs??????????? 
?

trainingp "Q?N
%?"
0??????????? 
%?"
1??????????? ?
8__inference_max_pooling_with_argmax2d_layer_call_fn_9580?I?F
/?,
*?'
inputs??????????? 
?

trainingp"Q?N
%?"
0??????????? 
%?"
1??????????? ?
L__inference_max_unpooling2d_1_layer_call_and_return_conditional_losses_10376?n?k
d?a
[?X
*?'
inputs/0?????????  @
*?'
inputs/1?????????  @

 
? "-?*
#? 
0?????????@@@
? ?
1__inference_max_unpooling2d_1_layer_call_fn_10382?n?k
d?a
[?X
*?'
inputs/0?????????  @
*?'
inputs/1?????????  @

 
? " ??????????@@@?
L__inference_max_unpooling2d_2_layer_call_and_return_conditional_losses_10579?n?k
d?a
[?X
*?'
inputs/0?????????@@@
*?'
inputs/1?????????@@@

 
? "/?,
%?"
0???????????@
? ?
1__inference_max_unpooling2d_2_layer_call_fn_10585?n?k
d?a
[?X
*?'
inputs/0?????????@@@
*?'
inputs/1?????????@@@

 
? ""????????????@?
L__inference_max_unpooling2d_3_layer_call_and_return_conditional_losses_10782?r?o
h?e
_?\
,?)
inputs/0??????????? 
,?)
inputs/1??????????? 

 
? "/?,
%?"
0??????????? 
? ?
1__inference_max_unpooling2d_3_layer_call_fn_10788?r?o
h?e
_?\
,?)
inputs/0??????????? 
,?)
inputs/1??????????? 

 
? ""???????????? ?
J__inference_max_unpooling2d_layer_call_and_return_conditional_losses_10173?p?m
f?c
]?Z
+?(
inputs/0??????????
+?(
inputs/1??????????

 
? ".?+
$?!
0?????????  ?
? ?
/__inference_max_unpooling2d_layer_call_fn_10179?p?m
f?c
]?Z
+?(
inputs/0??????????
+?(
inputs/1??????????

 
? "!??????????  ??
?__inference_model_layer_call_and_return_conditional_losses_8270?H()/012?@FGHIVW]^_`mntuvw????????????????????????B??
8?5
+?(
input_1???????????
p 

 
? "/?,
%?"
0???????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_8406?H()/012?@FGHIVW]^_`mntuvw????????????????????????B??
8?5
+?(
input_1???????????
p

 
? "/?,
%?"
0???????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_8855?H()/012?@FGHIVW]^_`mntuvw????????????????????????A?>
7?4
*?'
inputs???????????
p 

 
? "/?,
%?"
0???????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_9195?H()/012?@FGHIVW]^_`mntuvw????????????????????????A?>
7?4
*?'
inputs???????????
p

 
? "/?,
%?"
0???????????
? ?
$__inference_model_layer_call_fn_7095?H()/012?@FGHIVW]^_`mntuvw????????????????????????B??
8?5
+?(
input_1???????????
p 

 
? ""?????????????
$__inference_model_layer_call_fn_8134?H()/012?@FGHIVW]^_`mntuvw????????????????????????B??
8?5
+?(
input_1???????????
p

 
? ""?????????????
$__inference_model_layer_call_fn_9296?H()/012?@FGHIVW]^_`mntuvw????????????????????????A?>
7?4
*?'
inputs???????????
p 

 
? ""?????????????
$__inference_model_layer_call_fn_9397?H()/012?@FGHIVW]^_`mntuvw????????????????????????A?>
7?4
*?'
inputs???????????
p

 
? ""?????????????
"__inference_signature_wrapper_8515?H()/012?@FGHIVW]^_`mntuvw????????????????????????E?B
? 
;?8
6
input_1+?(
input_1???????????"E?B
@
activation_70?-
activation_7???????????