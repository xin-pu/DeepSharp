using RLSharp.Torch.Agents;

namespace RLSharp.Tests.RLTest
{
	public class AgentTest
	{
		public AgentTest(Agent agent)
		{
			Agent = agent;
		}

		public Agent Agent { get; set; }


		public float TestEpisode(int testCount)
		{
			var episode       = Agent.RunEpisodes(testCount);
			var averageReward = episode.Average(a => a.SumReward.Value);
			return averageReward;
		}
	}
}