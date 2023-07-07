using DeepSharp.RL.Agents;
using DeepSharp.RL.Environs;
using DeepSharp.RL.Trainers;

namespace TorchSharpTest.RLTest.PolicyBasedTest
{
    public class ReinforceTest:AbstractTest
    {
        public ReinforceTest(ITestOutputHelper testOutputHelper) 
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void TestReinforce()
        {
            var kArmedBandit = new KArmedBandit(new[] {0.4, 0.85, 0.7, 0.25}) {Gamma = 0.95f};
            var agent = new Reinforce(kArmedBandit, 16);
            var trainer = new RLTrainer(agent, Print);
            trainer.Train(0.90f, 500, "", 20, 2, false);
            agent.Save("Reinforce.st");
        }

        [Fact]
        public void VeirfyReinforce()
        {
            var kArmedBandit = new KArmedBandit(new[] {0.4, 0.85, 0.7, 0.25}) {Gamma = 0.95f};
            var agent = new Reinforce(kArmedBandit);
            agent.Load("Reinforce.st");
            var trainer = new RLTrainer(agent, Print);
            trainer.Val(20);
        }

        [Fact]
        public void TestReinforce2()
        {
            var frozenlake = new Frozenlake(new[] {0.8f, 0.1f, 0.1f}) {Gamma = 0.95f};
            var agent = new Reinforce(frozenlake, 16);
            var trainer = new RLTrainer(agent, Print);
            trainer.Train(0.95f, 500, "", 20, 2, false);
            agent.Save("ReinFrozen.st");
        }


        [Fact]
        public void verifyReinforce2()
        {
            var frozenlake = new Frozenlake(new[] {0.8f, 0.1f, 0.1f}) {Gamma = 0.95f};
            var agent = new Reinforce(frozenlake);
            agent.Load("ReinFrozen.st");
            frozenlake.ChangeToRough();
            var episode = agent.RunEpisode();
            Print(episode);
        }
    }
}
