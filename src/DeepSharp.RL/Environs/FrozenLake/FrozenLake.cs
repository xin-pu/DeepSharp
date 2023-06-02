using System.Text;
using DeepSharp.RL.Models;
using Action = DeepSharp.RL.Models.Action;

namespace DeepSharp.RL.Environs
{
    public class Frozenlake : Environ
    {
        public Frozenlake() : base("Frozenlake")
        {
            LakeUnits = new List<LakeUnit>();
            LakeUnits = CreateLake4();
            PlayID = 0;
            ObservationSpace = 16;
            ActionSpace = 4;
        }


        internal List<LakeUnit> CreateLake4()
        {
            foreach (var r in Enumerable.Range(0, 4))
            foreach (var c in Enumerable.Range(0, 4))
            {
                var unit = new LakeUnit(r, c, r * 4 + c);
                LakeUnits.Add(unit);
            }

            this[0].Role = LakeRole.Start;
            this[5].Role = this[11].Role = LakeRole.Hole;
            this[15].Role = LakeRole.End;

            return LakeUnits;
        }

        public int PlayID { set; get; }

        public List<LakeUnit> LakeUnits { get; set; }

        public LakeUnit this[int i]
        {
            get => LakeUnits[i];
            set => LakeUnits[i] = value;
        }

        public LakeUnit this[int r, int c]
        {
            get => LakeUnits[r * 4 + c];
            set => LakeUnits[r * 4 + c] = value;
        }


        /// <summary>
        ///     Change observation to Reward,
        ///     Only when at End Position will get reward:1
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public override Reward GetReward(Observation observation)
        {
            var index = observation.Value.argmax().item<long>();
            var unit = this[(int) index];
            switch (unit.Role)
            {
                case LakeRole.End:
                    return new Reward(1);
                default:
                    return new Reward(0);
            }
        }

        /// <summary>
        /// </summary>
        /// <param name="action">
        ///     0,1,2,3
        ///     0 => 上
        ///     1 => 下
        ///     2 => 左
        ///     3 => 右
        /// </param>
        /// <returns></returns>
        public override Observation UpdateEnviron(Action action)
        {
            var banditSelectIndex = action.Value.item<long>();
            var random = new Random();
            var p = random.NextDouble();
            var rowCurrent = this[PlayID].Row;
            var columnCurrent = this[PlayID].Column;
            p = 0.2;
            PlayID = banditSelectIndex switch
            {
                /// 往上走
                0 => p switch
                {
                    < 1f / 3 => this[new[] {0, rowCurrent - 1}.Max(), columnCurrent].Index,
                    < 2f / 3 => this[rowCurrent, new[] {0, columnCurrent - 1}.Max()].Index,
                    _ => this[rowCurrent, new[] {3, columnCurrent + 1}.Min()].Index
                },
                ///往下走
                1 => p switch
                {
                    < 1f / 3 => this[new[] {3, rowCurrent + 1}.Min(), columnCurrent].Index,
                    < 2f / 3 => this[rowCurrent, new[] {0, columnCurrent - 1}.Max()].Index,
                    _ => this[rowCurrent, new[] {3, columnCurrent + 1}.Min()].Index
                },
                ///往左走
                2 => p switch
                {
                    < 1f / 3 => this[rowCurrent, new[] {0, columnCurrent - 1}.Max()].Index,
                    < 2f / 3 => this[new[] {0, rowCurrent - 1}.Max(), columnCurrent].Index,
                    _ => this[new[] {3, rowCurrent + 1}.Min(), columnCurrent].Index
                },
                ///往右走
                3 => p switch
                {
                    < 1f / 3 => this[rowCurrent, new[] {3, columnCurrent + 1}.Min()].Index,
                    < 2f / 3 => this[new[] {0, rowCurrent - 1}.Max(), columnCurrent].Index,
                    _ => this[new[] {3, rowCurrent + 1}.Min(), columnCurrent].Index
                },
                _ => PlayID
            };

            var state = Enumerable.Repeat(0, LakeUnits.Count).ToArray();
            state[PlayID] = 1;
            var stateTensor = torch.from_array(state, torch.ScalarType.Float32);
            return new Observation(stateTensor);
        }

        public override float DiscountReward(Episode episode, float Gamma)
        {
            var res = (float) Math.Pow(Gamma, episode.Oars.Count);
            return res;
        }

        public override bool StopEpoch(int epoch)
        {
            return this[PlayID].Role == LakeRole.End || this[PlayID].Role == LakeRole.Hole;
        }

        public override void Reset()
        {
            base.Reset();
            PlayID = 0;
        }

        public override string ToString()
        {
            var str = new StringBuilder();
            foreach (var r in Enumerable.Range(0, 4))
            {
                foreach (var c in Enumerable.Range(0, 4))
                {
                    var lakeUnit = this[r, c];
                    var P = lakeUnit.Index == PlayID ? "P" : " ";
                    var detail = $"{lakeUnit} [{P}]";
                    str.Append(detail + "\t");
                }

                str.Append("\r\n");
            }

            return str.ToString();
        }
    }
}