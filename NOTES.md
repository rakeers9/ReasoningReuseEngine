
- Near misses hit higher than rewardings on the threshold, so we need to check that. - today
- We need to work on making the process faster, cheaper, and better aligned. - today
- We won't have the transcript in real use because the models are going to think out loud. We need to find a way to get that. - today
- We need a mechanism to decide if a task is cache-worthy or not. - We need to remove golden label. - today - these go together, as the correftness should be determined by the backlog of responses


- We need to productize this and work on the compilation step. See how compilation would handle if a function gets updated later or something like that, and we need a lot of testing, of course.
- More to look at: valid ranges are never enforced for parameters. Wrong answers are always silent, so if we get a wrong answer, we're not checking if that's actually wrong or not. We need to handle cache collisions. If multiple different functions show similarity, then how do we handle that? 
- Long-term extra use would be something where an LLM could still call some of these functions because they're already written out and ready to use. It'd be a waste if the LLM couldn't just invoke them and use them. Some way for the LLM to still use some of these functions that are written out, even if it's not a perfect hit for an adjacent task 



The one gap left

  If the model is wrong in the same way every time and users never notice, nothing above catches it, because
  every source of evidence agrees on the wrong answer. That's rare, but it's most likely exactly where users
  can't check for themselves: medicine, law, finance. For those topics, I'd add an occasional audit of cached
  entries by a stronger model or a person before the cache serves them widely.