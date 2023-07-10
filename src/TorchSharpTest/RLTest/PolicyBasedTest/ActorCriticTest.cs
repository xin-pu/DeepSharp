using DeepSharp.RL.Agents;
using DeepSharp.RL.Environs;
using DeepSharp.RL.Trainers;

namespace TorchSharpTest.RLTest.PolicyBasedTest
{
    public class ActorCriticTest : AbstractTest
    {
        public ActorCriticTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void ACKMTest()
        {
            var kArmedBandit = new KArmedBandit(new[] {0.4, 0.85, 0.7, 0.25});
            var agent = new ActorCritic(kArmedBandit, 4, gamma: 0.99f);
            var trainer = new RLTrainer(agent, Print);
            trainer.Train(0.90f, 300, testEpisodes: 20, testInterval: 2, autoSave: false);
            agent.Save("ACKM.st");
        }

        [Fact]
        public void ACKMCVal()
        {
            var kArmedBandit = new KArmedBandit(new[] {0.4, 0.85, 0.7, 0.25});
            var agent = new ActorCritic(kArmedBandit, 16);
            agent.Load("ACKM.st");
            var trainer = new RLTrainer(agent, Print);
            trainer.Val(20);
        }

        [Fact]
        public void ACFLTest()
        {
            var frozenlake = new Frozenlake(new[] {0.8f, 0.1f, 0.1f});
            var agent = new A2C(frozenlake, 16);
            var trainer = new RLTrainer(agent, Print);
            trainer.Train(0.90f, 600, testEpisodes: 20, testInterval: 2, autoSave: false);
            agent.Save("A2CFL.st");
        }

        [Fact]
        public void ACFLCVal()
        {
            var frozenlake = new Frozenlake(new[] {0.8f, 0.1f, 0.1f});
            var agent = new A2C(frozenlake, 16);
            agent.Load("A2CFL.st");
            var episode = agent.RunEpisodes(10);
            episode.ToList().ForEach(Print);
        }
    }
}