2022/10/27

*This file focuses on intro, overview and related work of each selected paper on the topic of motion generation. Experiments/methods/novelty will not be examined in this note. I tried to extract the authors' opinions towards other works and their thoughts on their works. Most words quoted are from the paper itself. The comment in Chinese is based on my own understanding regarding the work, and may not be neutral/correct.*  


# MotionCLIP: Exposing Human Motion Generation to CLIP Space

### Summary
Train an encoder to find the proper embedding of an input sequence in CLIP space, and a decoder that generates the most fitting motion to a given CLIP space latent code.
![[MotionCLIP.png]]
### Related Work
#### Guided Human Motion Generation
**1.1** Condition on another domain: ACTOR, Action2Motion. 
Music Conditioned 3D Dance Generation with AIST++ and Rhythm is a Dancer: Music-Driven Motion Synthesis with Global Structure:  conditions on other domains. 
**1.2** using another motion: motion style transfer
#### Text to Motion
**2.1** Current text-motion heavily based on KIT.  *Plappert et al. [2018] learned text-to-motion and motion-to-text using seq2seq RNN-based architecture. Yamada et al. [Yamada](https://ieeexplore.ieee.org/abstract/document/8403309) learned those two mappings by simultaneously training text and motion auto-encoders while binding their latent spaces using text and motion pairs. Lin et al. [2018] further improved trajectory prediction by adding a dedicated layer. Ahuja et al. [2019] introduced JL2P model, which got improved results with respect to nuanced concepts of the text, namely velocity, trajectory and action type*
**2.2** 提到了BABEL 数据集：是一个per-frame的textual labeles ordered in 260 classes to the larger AMASS dataset, including 40 hours of motion capture. 

~~## Action-Conditioned 3D Human Motion synthesis with Transformer VAE
Sign Language Production (SLP), as an essential task for the Deaf community, aims to provide continuously sign videos for spoken language sentences. 感觉和工作相关性不是很高，决定不看了，是一个使用手语系统生成pose的工作，主要重点是建立在G2P系统（因为手语系统和自然语言系统的区别） 
不过这篇工作的related work写的很详细。


# TEMOS: Generating diverse human motions from textual descriptions
"Generate 3D human motions from textual descriptions. This challenging task requires joint modeling of both modalities: understanding and extracting useful humancentric information from the text, and then generating plausible and realistic sequences of human poses." 
"Most previouos work focused on generating motions conditioned on a single action label, not a sentence", 这里提[[#guided-human-motion-generation]]到了Action2Motion[Link](#guided-human-motion-generation) 以及上面那篇Action-conditioned 3D human motion synthesis with transformer VAE。 
*Generating animated videos of human activities from natural language descriptions*, *Natural language grounded pose forecasting, Synthesis of compositional animations from textual descriptions*, 以上三篇被批评为：Generate only one output motion per text input, 

批评了RNN：On human motion prediction using recurrent neural networks，这个我觉得蛮无厘头的，找了一个2017年的RNN去作对比，现在谁不用全局信息？更多的是为了体现自己使用了Transformer做出的对比把。
![[TEMOS.png]]

### Summary
"We present Text-toMotions (TEMOS), a novel cross-modal variational model that can produce diverse 3D human movements given textual descriptions in natural language. (ii) In our experiments, we provide an extensive ablation study of the model components and outperform the state of the art by a large margin both on standard metrics and through perceptual studies. (iii) We go beyond stick figure generations, and exploit the SMPL model for text-conditioned body surface synthesis, demonstrating qualitatively appealing results."

### Related Work

#### Human Motion Synthesis
Motion generation can be broadly divided into two categories: 
1. Unconstrained generation, which models the entire space of possible motions 
*Convolutional sequence generation for skeleton-based action synthesis, Perpetual motion: Generating unbounded human motion,Bayesian adversarial human motion synthesis *
2. Conditioned synthesis, which aims for controllability such as using music, speech, action and text, 这里针对多模态也举了几个例子，我认为和第一篇的根据不同domain生成的感觉是一样的，只是第一篇理解为不同domain，这一篇理解为有条件的生成。
继续批评unconstrained：没有限制所以没有ability to control the generation process. 
提到conditioned 生成可以继续被分类：
1. Deterministic
代表工作：
*Generating animated videos of human activities from natural language descriptions
Synthesis of compositional animations from textual descriptions
Language2Pose: Natural language grounded pose forecasting*
2. Probabilistic(此为本篇的类型)
代表工作：
*Text2Action: Generative adversarial synthesis from language to action
Dancing to music
Learning to generate diverse dance motions with transformer*

#### Text-conditioned motion generation

提到了两类的做法。一类是把text-motion做成一个seq2seq的machnie translation 任务（代表作text2action），另一个是跨模态的编码，将文字和动作编码到同一空间（代表作Language2Pose）*详细的代表作名单可以参见论文原文，不做赘述*。
有两个代表工作对于motion的表征进行了加工：*Learning a bidirectional mapping between human whole-body motion and natural language using deep recurrent neural networks
Paired recurrent autoencoders for bidirectional translation between robot actions and linguistic descriptions*
这两篇都是Robotics的论文。
说到了motion的表征，text2action又被批评了：该工作只对上半身建模，因为使用了半自动的方法，而其使用的数据集MSR-VTT对于下半身动作有些是模糊的（这也行）他们是这样处理的：*“They apply 2D pose estimation, lift the joints to 3D, and employ manual cleaning of the input text to make it generic”*

本文提到了大部分的state of art的Text-Motion工作都是deterministic，使用了共享的跨模态latent space方法。

文章提到了*Language2Pose：Synthesis of compositional animations from textual descriptions*这个工作和他们的工作有些类似，但是主要区别是：*“Our key difference is the integration of a variational approach for sampling a diverse set of motions from a single text.”*

# Human Motion Diffusion Model
开篇点评：
为什么动作生成是一个挑战性的任务：可能的动作的丰富性，人类对其敏感的感知（这个没懂）以及如何准确描述动作。
稍微点评curent work：low-quality的生成或者表达能力有限。
然后本文是一个classifier-free diffusion-based generative model。本工作基于transformer，并且没有使用DDPM里的noise而是prediction of the sample（这里感觉开始有意思了）声称本工作是轻量模型（在实验部分提到了他们只用了单卡训练）且达到了SOTA效果（详情见文章实验对比）。
![[HumanMotionDiffusionModel.png]]
### Summary
提到了一个关于数据label的问题。动作“踢”，可以使soccer kick，也可以是Karate kick。指定动作kick，有很多种方法去描述他，所以成为了一个many to many的问题。提到了三篇工作以及他们的不足（都是使用了自编码器或者变分自编码器）：
*TEMOS: Generating diverse human motions from textual descriptions（就是上边这一篇）Motionclip: Exposing human motion generation to clip space
Language2pose: Natural language grounded pose forecasting（这个工作同时也被TEMOS提到了他们有较高的相关性我觉得有时间可以看一下。）

在HumanML3D和KIT数据集上达到SOTA

### Related Work
**（这个工作写的相关工作非常多而且很新）**
#### HUMAN MOTION GENERATION
总的来说本工作把之前的人类动作生成分成了两部分：
1. 有condition的生成，这个condition可以多种多样的（**其实就是不同domain map到了motion，同第一篇文章的分类**），比如有的是通过prefix pose（前缀动作）进行motion prediction （*Back to mlp: A simple baseline for human motion prediction
On human motion prediction using recurrent neural networks*）有的是“in-betweening and super-resolution tasks using bi-directional GRU”。还有一个工作通过自编码器学习到动作的隐式表达从而使用空间约束动作的生成。动作还可以被高层次的guidance指导，比如动作类别，声音还有语言（文字）。
2. 近年来的趋势是使用shared latent space for language and motion. 代表工作很熟悉了，JL2P，TEMOS， T2M，以及MotionCLIP（使用shared text-image latent space learned by CLIP）
#### DIFFUSION GENERATIVE MODELS
~~这一部分的工作没啥好讲的，都是Diffusion的介绍，和Motion generation没什么相关性。~~

# FLAME: Free-from Language-based Motion Synthesis & Editing
开篇点出几篇文章的问题：
*ActFormer: A GAN Transformer Framework towards General Action-Conditioned 3D Human Motion Generation，
Actionconditioned 3d human motion synthesis with transformer VAE（去掉的第二篇）
Action2motion: Conditioned generation of 3d human motions*
他们的问题是用了behavioral labels，但是这种标签会限制描述的能力，生成动作的多样性和对整体生成动作的可控性不够强。
然后是两篇提到的使用了近期的大规模预训练语言模型的论文：
*Synthesis of Compositional Animations from Textual Descriptions
TEMOS: Generating diverse human motions from textual descriptions（又是你）*
其中前者用了BERT，后者用了DistilBERT。
以上两篇论文作为具有出色性能代表的text-motion论文，作者对他们的批评是：
”they lack capability in a flexible conditional generation.“
文章的创新点：
”We propose FLAME, a unified model for motion synthesis and editing with free-form language description. • Our model is the first attempt applying diffusion models to motion data; to handle the temporal nature of motion and variable-length, we devise a new architecture. • We show FLAME can generate more diverse motions corresponding to the same text. • We demonstrate FLAME can solve other classical tasks—prediction and in-betweening—through editing, without any fine-tuning.“
（额，有点不懂的是居然在Intro里提到了他们的输入是什么）
![[FLAME.png]]
### Summary
一个Diffusion-based motion synthesis and editing model(上面那个也能editing我没提到)。结构里也用到了transformer，因为motion是variable-length的数据。
总的来说，本工作提出了一个transformer decoder-based architecture, which takes diffusion time-step token(as well as language token, motion token and motion lengths)

### Related Work
#### Diffusion Model & Text-Conditional Generation
前面一大堆话讲Diffusion Model, 甚至把起源的2015年那篇文章提出来了，这应该是这几篇用到Diffusion里面讲的最细致的了。然后就是介绍了几个Diffusion model在text generation的工作，包括：
*Diffusion models beat gans on image synthesis,
Glide: Towards photorealistic image generation and editing with textguided diffusion models*
前者是class conditional model，后者是text conditional model。

还提到了几个工作：
*Improved denoising diffusion probabilistic models， 对DDPM的改进，learning the reverse-diffusion variances
Classifier-free diffusion guidance, 是conditional generation without the need for a separate classifier model*
#### 3D Human Motion Generation
在介绍text-motion generation之前，提到了motion prediction models 和 in-betweening models（文献不一一列举了），他们的问题是根据one frame or given frames 进行预测，所以没有synthesize motion from textual descriptions.

然后开始讲text-motion的工作，以时间线为轴，最早的text-motion工作是2018年的
*Generating Animated Videos of Human Activities from Natural Language Descriptions和 Text2action: Generative adversarial synthesis from language to action*然后就是
*Action2motion: Conditioned generation of 3d human motions 和 
Actionconditioned 3d human motion synthesis with transformer vae*
最新的工作不再是text label而是free-form的text：
*Synthesis of compositional animations from textual descriptions 和
TEMOS: Generating diverse human motions from textual descriptions*
但是，这两篇工作的不足：“these models have limitations in extensibility to conventional motion tasks or text-based motion editing”


# MotionDiffuse: Text-Driven Human motion Generation with Diffusion Model

提出当前的text-motion generation 工作存在的问题：diverse and fine-grained motion generation with various text inputs. 声称他们的工作是第一个diffusion based text-driven motion generation framework. 
本工作主要贡献：
“Probabilistic Mapping. Instead of a deterministic language-motion mapping, MotionDiffuse generates motions through a series of denoising steps in which variations are injected. 2) Realistic Synthesis. MotionDiffuse excels at modeling complicated data distribution and generating vivid motion sequences. 3) Multi-Level Manipulation. MotionDiffuse responds to fine-grained instructions on body parts, and arbitrary-length motion synthesis with time-varied text prompts.”
![[MotionDiffuse.png]]
### Summary
有几类condition signals，包括pre-defined motion categories( *Action2Motion,
Actionconditioned 3d human motion synthesis with transformer vae,
Implicit neural representations for variable length human motion generation* ), music pieces 和nlp( *Synthesis of compositional animations from textual descriptions,
* )
对*TEMOS*的批评：”it does not support stylizing the generated motions and, therefore, could not achieve high diversity“
对*MotionCLIP*的批评："it is still limited to short text inputs and fails to handle complicated motion descriptions."
对两者的批评：“only accept a single text prompt, which greatly limits users’ creativity.”
与最初的*DDPM*对比：“Unlike classical DDPM which is only capable of fixed-size generation, we propose a Cross-Modality Linear Transformer to achieve motion synthesis with an arbitrary length depending on the motion duration.”

与MotionCLIP的对比：Instead of learning a direct mapping between the text space and the motion space (Tevet et al., 2022), we propose to guide the generation pipeline with input texts softly, which could significantly increase the diversity of the generation results

### Related Work
**这篇的相关工作写的非常详细！**
#### Motion Generative Model

以时间为轴，一开始是Statistical models such as PCA, Motion Graph
然后开始介绍与VAE， GAN， FLow-based，以及Implicit Neural Representations
最后提出了Diffusion相关工作。其中提到了VAE模型里面，近期耳熟能详的Motion Generation工作基本都是属于此类比如*MotionCLIP, ACTOR, Avatarclip: Zero-shot text-driven generation and animation of 3d avatars 以及TEMOS*

#### Conditional Motion Generation

作者认为 *Action-Conditioned 3D Human Motion synthesis with Transformer VAE， Action2Motion，Implicit neural representations for variable length human motion generation 等工作*   aim at synthesizing motion sequences of several specific categories。顺便还写了对Action2Motion，ACTOR的点评。
之后提到了music to dance（不是本文关注的重点了），不过也提到了三篇文章，大部分的思路是将music feature 和motion feature1 map到同一个joint space。只有一个工作是使用了two-stage dance generator。
在music-drive generation之后，作者提到了text-driven motion generation。 作者认为text-driven可以当做是"learning a joint embedding of text feature space and motion feature space". 当然，text和music肯定还有很大不同的，作者提到了两点不同：1. 文字驱动的模型与人体相关性更强 2. 文字驱动的模型包括了更广泛的动作。
作者认为：*Synthesis of compositional animations from textual descriptions 和 MotionCLIP*等工作还是以deterministic 模型为基础，因此他们只能生成一种动作，根据文字。TEMOS使用了VAE，可以生成不同的动作序列。但是这种动作序列是通过获取一个动作与语言的联合编码空间获得的，