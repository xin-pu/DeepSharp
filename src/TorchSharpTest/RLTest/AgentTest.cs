using DeepSharp.RL.Agents;

namespace TorchSharpTest.RLTest
{
    public class AgentTest
    {
        public AgentTest(Agent agent)
        {
            Agent = agent;
        }

        public Agent Agent { set; get; }


        public float TestEpisode(int testCount)
        {
            var episode = Agent.RunEpisodes(testCount);
            var averageReward = episode.Average(a => a.SumReward.Value);
            return averageReward;
        }
    }
}