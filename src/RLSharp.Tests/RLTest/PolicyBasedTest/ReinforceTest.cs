using RLSharp.Torch.Agents.Deep.Policy;
using RLSharp.Torch.Environs;
using RLSharp.Torch.Trainers;

namespace RLSharp.Tests.RLTest.PolicyBasedTest
{
	public class ReinforceTest : AbstractTest
	{
		public ReinforceTest(ITestOutputHelper testOutputHelper)
			: base(testOutputHelper)
		{
		}

		[Fact]
		public void ReinforceKMTest()
		{
			var kArmedBandit = new KArmedBandit(new[] { 0.4, 0.85, 0.7, 0.25 });
			var agent        = new Reinforce(kArmedBandit, 16);
			var trainer      = new RLTrainer(agent, Print);
			trainer.Train(0.90f, 500, testEpisodes: 20, testInterval: 2, autoSave: false);
			agent.Save("ReinKM.st");
		}

		[Fact]
		public void ReinforceKMVal()
		{
			var kArmedBandit = new KArmedBandit(new[] { 0.4, 0.85, 0.7, 0.25 });
			var agent        = new Reinforce(kArmedBandit);
			agent.Load("ReinKM.st");
			var trainer = new RLTrainer(agent, Print);
			trainer.Val(20);
		}

		[Fact]
		public void ReinforceFLTest()
		{
			var frozenlake = new FrozenLake(new[] { 0.8f, 0.1f, 0.1f });
			var agent      = new Reinforce(frozenlake, 16);
			var trainer    = new RLTrainer(agent, Print);
			trainer.Train(0.95f, 500, testEpisodes: 20, testInterval: 2, autoSave: false);
			agent.Save("ReinFrozen.st");
		}


		[Fact]
		public void ReinforceFLVal()
		{
			var frozenlake = new FrozenLake(new[] { 0.8f, 0.1f, 0.1f });
			var agent      = new Reinforce(frozenlake);
			agent.Load("ReinFrozen.st");
			frozenlake.ChangeToRough();
			var episode = agent.RunEpisode();
			Print(episode);
		}
	}
}