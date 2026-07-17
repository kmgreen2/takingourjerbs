# Pouring the Concrete Before the Blueprint: The AI build-out is priced on capabilities the architecture doesn't have

> **Disclaimer:** The purpose of this write-up is to give people a general understanding of the LLM-based AI landscape.
> There are concepts (e.g., cross-entropy loss and backpropagation), optimizations (e.g., the use of batching and the KV
> Cache), and mathematical details (e.g., the output of a model evaluation is a probability distribution, not a token)
> that I am leaving out for simplicity. I feel that digging too much into these details will take away from presenting the
> conceptual reality.  If you feel the need to "well actually" some minor technical detail, this
> isn't for you.  On the other hand, if you think something is conceptually wrong, I will verify, fix and update the text.

> **Note**: I am also probably wrong about some stuff.  This is a living document and is why I am hosting this as a README in Github.

> **TL;DR**: It's long, but here's the argument in five points:
>
> 1. **An LLM is a single stateless function**: It takes a sequence of tokens and returns the next one, probabilistically,
>     based entirely on that sequence.  All the state lives in a finite context window, not the
>     weights. When pre-training scaling laws hit diminishing returns, the industry
>     retrained the function to effectively work as a looping node in a state machine. The capability gain is real,
>     but it lives in the weights and only cashes out inside the loop.
> 2. **Errors compound**: Loops compound errors.  You can backtrack to mitigate errors, which works well with a cheap deterministic oracle, but it
>      is generally unclear how well we can mitigate compounding errors in use cases that do not have this type of verification.
> 3. **The "killer app" has cheap automated oracles**: Code is the killer agentic app because it has a compiler, a test
>     suite, a runtime that crashes. That's a property of the domain, not the underlying model. The industry is
>     extrapolating this success to all white-collar work, which has no such oracles.  Emails don't compile. And there's
>      no test suite for the fact that Chad is a pain to work with.
> 4. **Weights are not a moat**: The models are stateless, so the value is in managing the context and tools around the
>      model.  This means switching costs are proportional to managing context, tools and routing.  This will become
>      more important as we see more specialized local models, where inference can happen on a laptop, and where users
>      can decide to route on-demand based on price, domain or whatever.
> 5. **The build-out is priced as though 1-4 are not true**: A trillion dollars a year only pays back if agents get
>     reliable enough to displace workers wholesale.  This also assumes that demand lands on hosted frontier models, and it
>     does so inside the depreciation window. (2) reliability stays capped outside domains with cheap verification,
>     and (3) says most white-collar work cannot be cheaply verified.  In (4), the cheap, high-volume tail
>     of whatever demand does materialize has every incentive to flee to small models on commodity hardware and never
>     touch the data centers being built for it. The tool is real. The build-out isn't priced on the tool.

---

## Introduction

Since the introduction of ChatGPT in 2022, we have seen an unprecedented amount of growth and interest in applying AI to
all aspects of our lives. From generating therapists to source code, the generative AI boom has impacted people faster
than just about anything in recent memory. This all comes at a cost. Training and deploying the trained models requires a
great deal of infrastructure: GPUs to run the calculations, data centers to house the GPUs, energy
infrastructure to power the data centers and water to cool the data centers. We are experiencing a chaotic rush to build
out these data centers as fast as possible. Some believe this build-out is premature, while others argue it is necessary
to fulfill the demand and maintain technical superiority. The goal of this essay is to argue that the scope of the
build-out might exceed the actual demand, due to fundamental limitations in the technology. That is not to say the
frontier generative AI products are not extremely useful and highly valuable. They have shown to be extremely useful
tools, especially when augmenting humans. The limit is in the ubiquity of reliable autonomous agents, which would be
required to justify the proposed multi-trillion dollar infrastructure build-out.

## The Technology at a High Level

Instead of diving into the weeds of neural network architectures or the calculus of machine learning, we can distill an
LLM down to a single mathematical function. For our purposes, think of an LLM as a massive network of floating-point
numbers (called weights) that start with random values and are optimized over time to act as a statistical expression:

$$f(x) \rightarrow y$$

To feed text into this function, the input, $x$, must first be broken down into chunks called tokens, where a word is
chopped into one or more numeric pieces. These tokens are then mapped to numbers that describe their semantic meaning
using a process as embedding. For clarity and simplicity, we will ignore the distinction between tokens and words
throughout our examples.

Anyway, think of the LLM as a function $f(x)$ that takes a sequence of tokens $x$ and returns a single next token, $y$,
that is probabilistically chosen based entirely on that preceding sequence. The model is trained by iteratively updating
its internal weights using text. In a single training step, we take a sequence of text, remove the very last token to
create an incomplete sequence ($x'$), and then compute $f(x') \rightarrow y$.

The internal weights are updated according to the error, which is the mathematical delta between the model's guess ($y$) and the
actual token we removed. We repeat this process trillions of times across as much human text as possible, adjusting the
weights until the function effectively learns the statistical relationships of words in a sequence. This process
basically produces magic.  The foundation of even the most advanced modern models fits this conceptual paradigm.

> **Training Example:**
> * **Target Sequence:** "Why did the chicken cross the road"
> * **Incomplete Input ($x'$):** "Why did the chicken cross the"
> * **Model Prediction ($y$):** "coop"
>
>
> The internal weights are updated based on the $error(\text{"road"}, \text{"coop"})$ to "teach" the function the statistically correct word in this specific context.

### Context Windows

The **context window** is the hard limit on the maximum size of the input sequence, $x$, that can be fed into $f(x)$ at
any one time. Modern frontier models have context windows spanning anywhere from hundreds of thousands to multiple
millions of tokens.

If the size of your conversation or dataset exceeds this mathematical ceiling, the system is forced to cut down the data
to fit. This truncation is typically handled in one of a few ways: the application simply gives you an error and tells
you "tough shit, start over," it aggressively crops out the oldest chunks of the conversation, or the backing system
tries to summarize the existing history into a more compressed sequence to make room. No matter the engineering
workaround, exceeding the window means losing fidelity.

> Sidebar: I have become accustomed to the tell-tale groan that results from Claude telling you it has to compact the
> context. It is not uncommon for performance to degrade (i.e. forget things you just discussed), and in some cases, it
> is best to just start over.

### The First Use Case: Turn-Based Chat

We started with GPT-like chatbots that were used to have turn-based interactions between a person and an LLM. In fact,
this is how most people still interact with LLMs. Here is how turn-based interactions work. A human sends a question,
statement, rant or whatever to the LLM. This is tokenized into $x$. A process will generate a new token using $f(x)$,
appending the resulting token to the sequence at each step. The updated sequence is fed into $f(x)$ at each step. The
process stops when the resulting token is a special `STOP` token. The result is decoded from tokens into words in the
original language and sent to the user. Since this is an iterative process, we can stream token-by-token back to the
user.

Each turn results in the sequence getting larger and larger. For example, assume an initial query, $x_0$, and initial
response $r_0$. The second turn from the human will send $concat(x_0, r_0, x_1)$ to the LLM, which will result in $r_1$.
The third turn will send $concat(x_0, r_0, x_1, r_1, x_2)$ and so on. If the concatenation exceeds the context window,
the backing system can either return an error or it can try to compact the sequence potentially pissing the user off as
we discuss above.

> **Turn-Based Chat Example:**
> * **Human Initial Request:** "Tell me a joke!"
> * **LLM Response:** "Why did the chicken cross the road?"
> * **Human Response:** "That is a dumb joke"
> * **LLM Response:** "You hit the nail on the head and are so right!"
>
> Final evaluation: $f($`<human>`Tell me a joke`</human><ai>`Why did the chicken cross the road?`</ai><human>`That is a dumb joke`</human>`"}$)$ $\rightarrow$ "You hit the nail on the head and are so right!"
>

I hope this example shatters any anthropomorphic illusion you may have about LLMs. There is no memory in the traditional
sense. In fact the interaction with the weights is completely stateless, since all the state is maintained in the
context. Imagine how ridiculous the usual "round of intros" would be in a corporate setting if this was how humans interacted
with each other!?!?!

> **WTF Meeting Example:**
> * **Person 1 (Bob):** "Hi everyone, I’m Bob."
> * **Person 2 (Alice):** "Hi everyone, I’m Bob. And I’m Alice."
> * **Person 3 (Charlie):** "Hi everyone, I’m Bob. And I’m Alice. And I’m Charlie."
> * **Person 8 (Hannah):** "Hi everyone, I’m Bob. And I’m Alice. And I’m Charlie. [...six lines of repetition omitted for sanity...]. And I’m Hannah. The server is down."
>
>
> Final evaluation required just to hear from Hannah: $concat(\text{Bob}, \text{Alice}, \text{Charlie}, \text{Dave}, \text{Ellen}, \text{Frank}, \text{Grace}) \rightarrow$ "The server is down."

### The Scaling Laws

Many eons ago in the early 2020s, the industry operated under an unintuitive assumption: more data, more compute, and more
weights would lead to proportionally mo' smarter models. This was the era of the initial scaling laws.

Unfortunately, the tech giants quickly ran into two massive challenges. First, high-quality human training data is
finite, and we have already trained these models on virtually the entire internet. Second, the labs hit
diminishing marginal returns. They discovered that achieving linear improvements in model performance required
exponential increases in data, parameter size, and computing power (this was formalized by the Chinchilla scaling
laws).  It's not that we hit a hard ceiling, but the additional gains are simply not worth the cost.

At this point, the industry was left with three choices: have the snake eat its own tail by training new models on
flawed synthetic data, wait for an unimaginable amount of new computing power to magically become available, or figure
out a way to leverage the context window to make the model "think."

Yup, we chose the third option. We ended up telling the models to have a ridiculous corporate meeting with themselves
before giving the human an answer.

### Mitigating the Scaling Laws: The Rise of the State Machine

To bypass the flattening curves of pre-training, the industry had to pivot. If we couldn't make the base function $f(x)$
exponentially smarter by simply dumping more data into its weights, we had to change *how* we executed the function.
This mitigation strategy gave rise to both Chain-of-Thought (CoT) reasoning and autonomous agentic loops.

At a high level, CoT and agents are cut from the exact same architectural cloth. Both are state machines that use the
rolling context window as their current operational state at each step.

I hope something is clear now. Much of the impressive AI breakthroughs we have witnessed over the last few years have
not been driven by a change in the underlying mechanism. The core next-token prediction remains the identical. The key
is that the weights have been trained, via reinforcement training (RL), to drive agentic loops, which allows the models to
self-reason and call external tools when they see fit. The engineering leap has been in the state machines build on top
of them and retraining the underlying model to operate as part of the state machine.

### Chain-of-Thought

Ok, let's talk about chain-of-thought. Instead of taking the human query $x$ and immediately generating the first token
of the visible response $r_0$, the model leverages $f(x)$ to iterate with itself in a hidden scratchpad. It generates a
long sequence of internal "thinking" tokens, appending each back into $f(x)$ exactly like turn-based chat, but is
typically obscured from the user. This internal loop allows the model to map out logic, evaluate intermediate
mathematical steps, and catch its own errors entirely within its own weights before committing to a visible answer. Only
when the model decides this internal process is complete does it transition to generating the final response sequence
for the user. This effectively converts temporal compute into higher-quality token predictions, allowing the model to "
think things through" before it speaks. This reasoning is trained by having the model produce reasoning attempts, where
productive attempts are rewarded and unproductive attempts are discouraged. This form of reinforcement learning will
update the underlying weights to encourage the model to do "productive reasoning."

### Agents

In its most basic form, an agent is basically a loop, a set of tools and an LLM. That is, given an objective, an agent
will iterate with itself to reach the objective. Instead of Human-to-AI turns, an agent
allows `Human-to-AI-to-AI-to..-AI-to-Human loops`. Assume a human sends a task to an agent. At each turn with itself,
the agent appends to the sequence and evaluates $f(x)$. The result can tell the agent to use a tool or tools, such as
fetch a webpage, call an API, fetch a file or pretty much anything you can do on a computer. The results are appended to
the sequence and fed into $f(x)$. This process repeats until the context window limit is reached or the LLM decides it
is done. If the context window is reached, agents can throw an error or will compact the sequence by summarizing and/or
getting rid of older tokens in the sequence. Think of the context window as the agent's short term memory and tools as
methods of discovery or access to the agent's long-term memory.

The main difference between CoT and agentic loops is that, in CoT, the state is updated using only direct results from
$f(x)$ (i.e. model weights), while agents update state using both $f(x)$ and external data via tool calls. This allows agents
to access long-term memory and fetch up-to-date information via API calls, etc.

While I make a distinction between CoT and agents, note that modern models often interleave tool calls with reasoning,
blurring the two approaches.

### The Foundational Catch

Before moving on, I have to admit a few things. First, while the conceptual foundations of these models are relatively
simple, the mathematics and engineering required to reliably deploy them is super complex. Given the sudden growth, I am
amazed that these companies can keep things from constantly falling over. Sure, a status page showing one-to-two 9s
isn't great, but I'd like to see you run the infrastructure for a company experiencing unprecedented growth during a
hardware shortage. Second, the current state-of-the-art is responsible for some very remarkable results. My day job as a
software engineer is completely different today than it was 9-12 months ago. I literally write code through Claude Code,
Codex and `goose`, and do not see a world in which a software engineer doesn't do the same. All of this said, we still
do not know exactly why these models are as effective as they are. At this point, it is indistinguishable from magic.

Magic aside, there is a foundational catch to the current approach the industry is taking. No matter how many internal
reasoning scratchpads or external tools you stack on top of $f(x)$, you are still ultimately just looping a stateless,
probabilistic calculator. The catch is that RL only improves the function where something can grade its attempts
reliably and cheaply, as we see with the coding use case. Where a grader is ineffective or missing, tiny statistical
errors compound until the system can drift, break, or spin off into oblivion. Because this architecture is
optimized for what sounds plausible rather than what is objectively true, it may never cross the reliability threshold
needed to completely offload human labor. We are aggressively building out a trillion-dollar infrastructure for a sci-fi
future that may never come.

This reliability ceiling becomes obvious when you look at the data source. These models are trained almost entirely on
human-generated content.  Humans are deeply flawed. We lie, we make mistakes, and we post things on the internet for
shits and giggles. We also do great things and have surprising insights, but the model mirrors the garbage right along
with the genius. There is a dangerous, pervasive assumption in Silicon Valley that AI will eventually iterate its way
to "perfection," completely ignoring the reality that a statistical mirror can be no better than the data it is
reflecting.

This brings us to the industry's favorite new narrative, which is the claim that humans are transitioning from "creators" to "
reviewers." I agree to a certain extent, seeing I spend far more time reviewing generated output today than I did a year
ago. But this narrative ignores basic human psychology. Human nature dictates that we will always take the path of least
resistance. Engineers will get tired of auditing endless lines of probabilistic code, rubber-stamp the outputs, or
worse, deploy a secondary LLM to review the primary LLM. At that point, you are running generative AI workflows with
zero human oversight. I try really hard to review everything that crosses my desk, but I feel incentives will eventually win
out in a corporate setting.  Pressure to do more with AI coupled with the surface-level convincing nature of the output, will
likely lead to less oversight.  Sure in an ideal world, engineers will completely switch their workload to be more reviewer
than creator, but we are already seeing pressure to produce more, which likely means spending less time reviewing.

This systemic rubber-stamping can create a massive downstream "model collapse."  When lazy teams commit plausible but
subtly buggy generated code and unverified documentation directly to internal repositories and wikis, they contaminate
the enterprise's context. Within a couple of iteration cycles, the next generation of models is fine-tuned or RAG-ing on
the previous model's synthetic garbage, locking the organization into a self-degrading, error-propagating feedback loop.

Ok, enough of this tangential rant.  Let's try to reason about the potential outcomes in light of the underlying technological
limitations.

> Sidebar on Reliability: It is unclear if the poor reliability of the frontier models is a function of "vibe
> coding" (e.g. engineers write code through agents like Claude Code and Codex) demand in the face of hardware
> shortages, or the general insane pace of change in the space. Like most things, it is probably a combination of all of
> this.
>

## The Bull Cases: What Would Have to Be True

So far we've established that the main technology is chain-of-thought and agentic loops layered on top of a stateless,
probabilistic next-token predictor, where errors compound and increase token cost. The question this section takes up is
that the industry is committing on the order of a trillion dollars a year to infrastructure, and we want to reason about
what would have to be true for that build-out to pay off. I do not want to predict the future. I want to lay out the
bull cases honestly and see how a few outcomes measures up against the capital already committed.

### Messaging that Might Contradict the Committed Capex

First, before we go into the weeds on possible outcomes, it is important to highlight the current state of the AI boom
against the messaging used to justify the current capex. Recently, the big AI labs have shifted their messaging from
scaring the public with claims of AGI and human replacement to providing copilots/assistants with a human-in-the-loop.
The build-out is sized for a claim that we are on a path to AGI, which needs mo' compute. There are many possible
reasons the messaging has changed. For example, this could be a marketing pivot due to poor public opinion (people don't
like when you take their jobs) or the leaders realize the limitations and have decided to finally come clean. It is odd
that such a huge shift in messaging isn't affecting the build-out. Anyway, I'd rather not focus on why the messaging
changed and why it has not affected the capex spend. I just thought it would be interesting to mention it and will not
mention it again.

### A Number We Won't Pretend to Resolve

Much of the AI-related headlines highlight the capex the hyperscalers and frontier labs are committing to. Capital
expenditure is a number each company reports on its own audited statements.  For example, the four largest hyperscalers guided to
roughly \$700B in combined 2026 capex, nearly double their 2025
spend ([CNBC](https://www.cnbc.com/2026/02/06/google-microsoft-meta-amazon-ai-cash.html)) and Goldman Sachs estimates a
cumulative ~\$5.3T across the Big Four from
FY2025–FY2030 ([Yahoo Finance](https://finance.yahoo.com/sectors/technology/article/meta-microsoft-amazon-and-alphabet-are-about-to-spend-a-shocking-amount-of-money-to-dominate-the-ai-era-115359575.html)).
That is a lot of cheddar!

The revenue that is supposed to justify that spend is, well, a bit sketchy, and very hard to tease apart from the outside.
Consider a single transaction. When a company uses Anthropic's models through AWS Bedrock, that one stream of dollars is
claimed, in part, by more than one firm's reporting.  Anthropic books cloud-reseller sales on a gross basis, counting
the full end-customer spend as revenue with AWS's cut recorded as an expense,
while AWS also recognizes revenue on the same transaction. Widen the lens and it worsens. Amazon is at once Anthropic's
investor, its compute provider, and its reseller.  That is, capital invested into a lab flows back out as compute spend and
returns again as revenue when customers buy the models through the investor's own cloud. Bloomberg mapped this web of "
circular deals" in detail ([Bloomberg](https://www.bloomberg.com/graphics/2026-ai-circular-deals/)).  Fuck!  What a mess!

On top of the entanglement sits a timing problem. The headline revenue figures the labs disclose are run-rates. A
run-rate is the most recent month annualized, which, for companies whose usage is doubling in a matter of months, say
more about the latest spike than about what will actually book across the year. As far as I know, as of this writing, no
independently verifiable full-year revenue number that can be cleanly set against the capex.

So I am not going to adjudicate these figures, reconcile them, or build a total out of them. I will take the committed
capex as a stake in the ground and note that **the capital is being committed against a revenue base no one outside
these companies can actually measure.** I am not the first to flag the gap. Sequoia's David Cahn put the annual distance
between AI infrastructure spend and AI ecosystem revenue at roughly **\$600B**, and Allianz Research has argued the
capex-to-revenue divergence already exceeds that of the 2001 telecom
build-out ([Forbes](https://www.forbes.com/sites/jasonkirsch/2026/06/02/the-ai-capex-to-revenue-gap-is-widening---and-markets-are-starting-to-notice/)).

Two quick points before moving on, to make sure I don't get flak for ignoring depreciation and the general-use aspects
of the build-out. First, hardware has a depreciation schedule, usually between 3–6 years, so the cost of something
bought in 2026 is spread over the lifetime of that component. Second, the data center build-out includes the
infrastructure to host the GPU capacity needed to serve tokens, which can be re-purposed for any other cloud-like use
case. Both points cut against the "numbers don't add up" narrative. But I don't think either of these points closes the
gap, because the spend isn't sized to a rational number people can measure. It is sized to the risk that "the other guy" will get
there first. This is an arms race, where the spend stops when someone wins or runs out of money.

For the cases that follow, the capex is the fixed quantity and demand is the open question. Rather than assert a revenue
figure I have to defend, I try to build the demand side from the ground up. I built a simple model, and will present a
few potential outcomes. An interested reader can substitute their assumptions into the model [at the site](https://kmgreen2.github.io/takingourjerbs/). If you can
find a set of inputs that clears the committed line without assuming every variable breaks the same way at once, the
model will show it.

### Case 1: Software Augmentation

Software development has proven to be the killer app for generative AI. As I have discussed earlier, software
engineering has completely changed overnight. A big part, but not the only part, of the software development lifecycle
is creating and updating code. This aspect of the job has changed from an engineer writing code to an engineer iterating
with an agent using a mixture of code and plain english. Note that I use the word iterate. For many tasks, the agent
needs context and direction, which isn't always possible in one shot. Remember the underlying model is stateless, so the
human has to provide much of the context for a given task, and there is plenty of room for ambiguity or errors in this
context. In addition, agents can hallucinate or misinterpret instructions or motivations. This means there is still
plenty of work for the human to make sure the agent is doing the right thing. At each turn, the agent can quickly
generate good code. The human can look at it and provide feedback, where the agent might quickly update the code, and so
on. The efficiency here is the speed of generating syntactically correct, formatted, tested code. While the code likely
runs, is formatted nicely and tested, it doesn't mean the code is bug free or even matches what the user wants. This is
why the process is iterative. Since code generation is effectively free (or really cheap), engineers are finding it
easier to iterate with an agent that can generate hundreds of lines of code in seconds, rather than manually generate
those lines in hours.

Code is the killer agentic app, because it has independent, cheap, automated oracles: a compiler, a test suite, a
linter, a runtime that crashes or doesn't. The agentic loop functions because it can iterate against fast signals by
running tests, reading the error and retrying. Try to run the program, look at the output, retry. And so on. This is the
entire reason agentic coding is the leading use case, and it is a property of the domain, not of the model.

> Sidebar on Verification: Complete verification is possible in theory but never complete in practice. Its incompleteness is
> precisely where a probabilistic generator's errors hide. Tests verify the paths you and the agent thought to test.
> Types verify shape, not intent. The gap between "compiles and passes tests" and "does what was meant" is exactly the
> gap a plausibility-optimized model is built to slip through. Closing it would require knowing the intent and the
> intent lives in the human.

Let's assume universal adoption within software development and assume a \$18k/engineer/year cap, which is the cap Uber
announced in [Bloomberg](https://techcrunch.com/2026/06/02/uber-caps-employee-ai-spending-after-blowing-through-budget-in-four-months).
Given the median wage of
a [software developer](https://www.bls.gov/ooh/computer-and-information-technology/software-developers.htm) in the US is
\$133k, this accounts for roughly 13.5% of wages in tokens. This works out to:

```
1,895,000 × $18,000 = $34.1B / year
```

Assuming universal adoption at the highest price a big tech, AI-forward company is willing to pay, is about \$34B a year.
This means even total success in the one domain where AI unambiguously works tops out near 13.5% of a wage, not 100% of
it. Software alone cannot justify the build-out. We need a bigger slice of the labor market.

### Case 2: White-Collar Replacement

A market size of `$34.1B/year` is huge, but doesn't justify the capex. To justify the build-out, we need to disrupt a
larger part of the labor market. One way to do this would be to expand from software development to all white-collar
work.  Let's go crazy and even assume a large fraction of jobs can be replaced by AI agents.

```
Replacement revenue = (jobs automated) × (wage) × (agent price as fraction of wage)
```

Let's plug in some numbers to help us reason about a potential market size before exploring the reliability needed to
replace workers with agents. At total automation of the professional class, 25% of wage, where the median for management/professional roles is
about [$85k/year](https://www.bls.gov/news.release/wkyeng.htm) and there are a combined 70M workers across all white collar occupations:

```
70,000,000 × $85,540 × 0.25 = $1.5T / year
```

This easily clears the line and justifies the capex. There are a few high-level hurdles preventing us from replacing white-collar workers
with agents. These hurdles are reliability and intent.  We will cover additional limitations that also apply in the replacement case when discussing
white-collar augmentation below.

1. **The Reliability Math**: An agent making $N$ sequential model calls, each with independent failure probability $p$,
   succeeds at roughly $(1−p)^N$. For example, if we assume that a model provides the correct answer 99% of the time,
   given its current context, then we can expect a 25-step task is correct 78% of the time, a 50-step task is correct
   60% of the time and a 100-step task is correct about 38% of the time.  In reality, the success probability at each step is
   highly dependent on how previous steps modified the context, for better or worse.  With cheap, reliable oracles, errors can be
   detected and corrected at each step, so there are cases where the 38% example above would not happen.  That said, there are
   also cases where errors compound because they are not detected.  I use the example only as an illustration and will just
   point out that this can only be avoided with automated, reliable detection mechanisms.
2. **Intent**: Replacement doesn't merely require reliable execution. It requires **eliminating the human who supplies
   intent.** Every system in this essay is a function from context to tokens.  The context comes from a person who
   wants something. Full replacement means the objective itself must be generated *inside* the loop.

In summary, the full replacement case requires the underlying model be very reliable. This reliability is directly
dependent on quality and precise context, which is in part dependent on the intent. Humans still have to supply the
intent.

Replacement definitely justifies the build-out capex. But clearing it on paper requires agents reliable enough to run
hundred-step tasks without drifting, and an intent-generator that the architecture simply doesn't provide. Both are
load-bearing, and both are exactly the limitations the first half of this essay laid out.

### Case 3: White-Collar Augmentation

Ok, so complete replacement of white-collar work seems unlikely, let's reason though a middle ground that fits the
latest messaging from the AI companies. Let's look at white-collar augmentation.

There are roughly 70M white-collar jobs in the US, with average wages of \$85k/year.  Let's assume a similar token spend
as software development to get:

```
70,000,000 × $85,540 × 0.10 = $599B / year
```

This means if every one of the 70M American professionals gets 1/10th of their salary in tokens, we get very close to the
capex line.  While promising, let me explain why this outcome is not likely.

1. **Verification asymmetry**: In software, review is cheaper than writing. In most knowledge work, it isn't. A memo, a
   brief, a financial model, a diagnosis, a strategy deck have no compiler, no test suite, no runtime. When you don't
   have reliable, automated review, verification cost approaches production cost. When verification cost matches or
   exceeds original production cost, the economic surplus of automation collapses to zero. Instead of saving labor, you
   merely shift the human's role from a high-leverage "creator" to an exhausted, low-leverage "auditor." This
   verification bottleneck is a potential economic drag of agentic systems. This reality is formally modeled in
   a [2026 MIT paper](https://arxiv.org/abs/2602.20946), which outlines how biologically bottlenecked human verification
   bandwidth caps the ultimate value of cheap machine execution. This is the cheap, reliable grader problem from earlier landing on the
   revenue side.  That is, many other forms of work (including white-collar) do not have the fast oracles needed to quickly improve
   the technology.
2. **Token usage asymmetry**:  Coding agents iterate against a fast, free oracle. That loop is what makes them work.
   Knowledge work has no fast oracle.  In this case, the agent either loops without signal or terminates
   early and hands a plausible artifact to a human who must fully verify it. You get the token cost of agentic reasoning
   without the reliability benefit of the feedback loop.
3. **Context asymmetry**:  A repository is bounded, retrievable, textual. The context required to write a good strategy
   memo includes the last three off-sites, the CEO's temperament, the client's unstated politics, and the thing everyone
   knows but nobody wrote. That context is in no corpus and cannot be paged into a context window, because it was never
   externalized. The statelessness argument bites hardest here, since the model has no memory, and the state that matters was
   never written down to begin with.
4. **Adoption asymmetry**: Developers self-serve and adopt bottom-up. Enterprise white-collar adoption is procurement,
   compliance, training, and workflow redesign. Universal adoption at 10% of wages assumes an adoption curve no
   enterprise technology has ever achieved.  Furthermore, it assumes it happens inside the capex depreciation window.
5. **Regulated domains**: Medicine, law, audit, finance have liability regimes that require a named human to
   sign. Augmentation there is capped by statute, not capability.

$599B requires universal adoption, sustained forever, in domains that have none of the cheap verification that makes
software work, at a spend rate borrowed from the one field that does.

### Case 4: Decide for yourself

I have provided a website that allows you to model this for yourself: [Have fun](https://kmgreen2.github.io/takingourjerbs/)

### The Price Contradiction

Let's return to the replacement formula. Total automation of the entire professional class is one of the strongest cases
for a multi-trillion dollar build-out. The whole result hinges on the agent's price as a fraction of the wage it
replaces. I set it at 25%. Justifying the capex spend needs that number to stay high.

But every force in this essay pushes it down. The base model is a stateless function and all the state lives in the
context, not the weights. This means that moving a workload from one provider to another is a configuration change, not
a migration. If the durable engineering advantage is in the wrapper and not the model, then the model underneath is a
commodity bought on price. And commodities get bid toward marginal cost.

The hyperscalers know this. They realize weights aren't a moat, which is why they are aggressively trying to build a
physical, real-world utility moat. By locking up power grid capacity, nuclear energy contracts, fiber lines, and custom
silicon pipelines, they hope to construct high-barrier tollbooths. But this turns the AI boom from a high-margin
software play into a highly capital-intensive, slow-depreciation utility bet. This is not exactly the venture-scale
dream being priced in.

This is the contradiction at the center of the replacement case.  That is, the build-out needs agents cheap enough to displace a
worker and expensive enough to stay pricey while doing it. Cheap agents mean mass replacement and thin revenue.
Expensive agents mean fat revenue and slow replacement.

### The Demand Splits

Every case above granted the build-out its best possible routing, where all of this demand lands on hosted, closed,
frontier models in a hyperscaler's data center. Withdraw that assumption and the picture changes.

Because inference is stateless, the cheap tail of the work has nowhere it needs to stay. The high-volume, low-difficulty
tasks, such as classification, extraction, routing, summarization, etc. run acceptably on small open-weight models.
Furthermore, those can run on commodity hardware, on-premise, or on a laptop. The economics are not subtle.  Serving a
workload yourself, or on a cheap open-weight host, will likely undercut a frontier API once volume is steady. The moment
switching is a config change and the small model is good enough, the long tail leaves.  Some may argue that many companies
are not sophisticated enough to route to different models and host their own context, which could justify using the more
expensive off-the-shelf agentic tools from the big AI companies.  We are in the early innings, and if token cost is ultimately
a race to the bottom, the smart layers between users and the models will show up.

I am not arguing that demand collapses, I am arguing that demand will split. The heavy, high-value agentic work, the
complex coding and reasoning loops, concentrates in expensive centralized compute and stays there. The long tail of
simple and routine work flees to cheap, local inference and never touches the data centers being built for it. The trillion-dollar
build-out is sized as though all the demand grows, stays centralized, and monetizes on the capex timeline.

## Conclusion: The Tool Is Real, the Replacement Is the Bet

The technology underneath all of this is a stateless, probabilistic next-token predictor, made useful by wrapping it in
loops. It is genuinely transformative in domains with cheap, fast verification. Software is the killer app
not because the model is smart but because code has a compiler, a test suite, and a runtime that tell the loop when it's
wrong. Where those oracles exist, the tool is real and here to stay.

But a useful tool in a single domain does not justify a huge data center build-out. Total success in software is a
rounding error against the capex. A sea change like white-collar displacement, which is needed to justify the capex,
needs reliability the architecture can't reach and a price that stays high enough to fill the revenue while low enough
to justify firing the human. Large-scale white-collar augmentation, the plausible middle, still falls short, due to the
many asymmetries compared to the coding use case. And because inference is stateless, the cheap work has every reason to
flee the data centers the capex is building.

A trillion dollars a year isn't priced as a bet on augmentation, which is what the technology reliably delivers, and
what the labs have quietly retreated to selling. It's priced on universal adoption, sustained pricing, centralized
demand, and autonomous reliability all arriving at once, inside the depreciation window. The honest position isn't that
it won't eventually happen, it's that we're pouring the concrete as though it already has.
