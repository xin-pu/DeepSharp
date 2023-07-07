using DeepSharp.RL.Agents;
using DeepSharp.RL.Environs;
using DeepSharp.RL.Trainers;

namespace TorchSharpTest.RLTest.TrainerTest
{
    public class RLTrainTest : AbstractTest
    {
        public RLTrainTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void TrainCreateTest()
        {
            var kArmedBandit = new KArmedBandit(new[] {0.4, 0.85, 0.75, 0.75});
            var agent = new QLearning(kArmedBandit);
            var trainer = new RLTrainer(agent, Print);
            trainer.Train(0.9f, 1, "", 20, 2);
        }


        [Fact]
        public void TrainCallBackTest()
        {
            var kArmedBandit = new KArmedBandit(new[] {0.4, 0.85, 0.75, 0.75});
            var agent = new DQN(kArmedBandit, 100, 1000);
            var trainer = new RLTrainer(agent, Print)
            {
                Callback = new TestCallBack()
            };
            trainer.Train(0.9f, 500, "", 20);
        }


        private class TestCallBack : TrainerCallBack
        {
            public override void OnTrainStart()
            {
                RlTrainer.Print?.Invoke("Hello, this info comes from callback");
            }

            public override void OnTrainEnd()
            {
            }

            public override void OnLearnStart(int epoch)
            {
            }

            public override void OnLearnEnd(int epoch, LearnOutcome outcome)
            {
            }

            public override void OnValStart(int epoch)
            {
            }

            public override void OnValEnd(int epoch, Episode[] episodes)
            {
            }

            public override void OnSaveStart()
            {
            }

            public override void OnSaveEnd()
            {
            }
        }
    }
}