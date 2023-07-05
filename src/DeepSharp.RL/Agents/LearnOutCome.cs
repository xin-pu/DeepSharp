using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    public class LearnOutcome
    {
        public LearnOutcome()
        {
            Steps = new List<Step>();
            Evaluate = 0;
        }

        public LearnOutcome(Step[] steps, float evaluate)
        {
            Steps = steps.ToList();
            Evaluate = evaluate;
        }

        public LearnOutcome(Episode episode)
        {
            Steps = episode.Steps;
            Evaluate = episode.SumReward.Value;
        }

        public LearnOutcome(Episode[] episode)
        {
            Steps = episode.SelectMany(e => e.Steps).ToList();
            Evaluate = episode.Average(a => a.SumReward.Value);
        }

        public LearnOutcome(Episode[] episode, float loss)
        {
            Steps = episode.SelectMany(e => e.Steps).ToList();
            Evaluate = loss;
        }

        public List<Step> Steps { protected set; get; }
        public float Evaluate { set; get; }

        public void AppendStep(Step step)
        {
            Steps.Add(step);
        }

        public void AppendStep(IEnumerable<Step> steps)
        {
            Steps.AddRange(steps);
        }

        public void AppendStep(Episode episode)
        {
            Steps.AddRange(episode.Steps);
        }

        public void UpdateEvaluate(float evaluation)
        {
            Evaluate = evaluation;
        }

        public override string ToString()
        {
            var avrReward = Steps.Average(a => a.Reward.Value);
            var message = $"S:{Steps.Count}\tR:{avrReward:F4}\tE:{Evaluate:F4}";
            return message;
        }
    }
}