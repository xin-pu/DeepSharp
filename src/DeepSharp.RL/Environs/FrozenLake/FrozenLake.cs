using System.Text;
using DeepSharp.RL.Models;

namespace DeepSharp.RL.Environs
{
    public class Frozenlake : Environ
    {
        public Frozenlake(torch.Device device)
            : base("Frozenlake", device)
        {
            LakeUnits = new List<LakeUnit>();
            LakeUnits = CreateLake4();
            PlayID = 0;
            ObservationSpace = 16;
            ActionSpace = 4;
        }

        public Frozenlake(DeviceType device)
            : this(new torch.Device(device))
        {
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


        internal List<LakeUnit> CreateLake4()
        {
            foreach (var r in Enumerable.Range(0, 4))
            foreach (var c in Enumerable.Range(0, 4))
            {
                var unit = new LakeUnit(r, c, r * 4 + c);
                LakeUnits.Add(unit);
            }

            this[0].Role = LakeRole.Start;
            this[6].Role = this[11].Role = LakeRole.Hole;
            this[15].Role = LakeRole.End;

            return LakeUnits;
        }


        /// <summary>
        ///     Change observation to Reward,
        ///     Only when at End Position will get reward:1
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public override Reward GetReward(Observation observation)
        {
            var index = observation.Value!.argmax().item<long>();
            var unit = this[(int) index];
            switch (unit.Role)
            {
                case LakeRole.End:
                    return new Reward(1f);
                default:
                    return new Reward(0f);
            }
        }

        /// <summary>
        /// </summary>
        /// <param name="act">
        ///     0,1,2,3
        ///     0 => Up
        ///     1 => Down
        ///     2 => Left
        ///     3 => Right
        /// </param>
        /// <returns></returns>
        public override Observation UpdateEnviron(Act act)
        {
            var banditSelectIndex = act.Value!.item<long>();

            var moveProb = torch.multinomial(torch.from_array(new[] {1 / 3f, 1 / 3f, 1 / 3f}), 1);
            var moveAction = moveProb.item<long>();
            var rowCurrent = this[PlayID].Row;
            var columnCurrent = this[PlayID].Column;

            PlayID = banditSelectIndex switch
            {
                /// 往上走
                0 => moveAction switch
                {
                    0 => this[new[] {0, rowCurrent - 1}.Max(), columnCurrent].Index,
                    1 => this[rowCurrent, new[] {0, columnCurrent - 1}.Max()].Index,
                    2 => this[rowCurrent, new[] {3, columnCurrent + 1}.Min()].Index,
                    _ => throw new ArgumentOutOfRangeException()
                },
                ///往下走
                1 => moveAction switch
                {
                    0 => this[new[] {3, rowCurrent + 1}.Min(), columnCurrent].Index,
                    1 => this[rowCurrent, new[] {0, columnCurrent - 1}.Max()].Index,
                    2 => this[rowCurrent, new[] {3, columnCurrent + 1}.Min()].Index,
                    _ => throw new ArgumentOutOfRangeException()
                },
                ///往左走
                2 => moveAction switch
                {
                    0 => this[rowCurrent, new[] {0, columnCurrent - 1}.Max()].Index,
                    1 => this[new[] {0, rowCurrent - 1}.Max(), columnCurrent].Index,
                    2 => this[new[] {3, rowCurrent + 1}.Min(), columnCurrent].Index,
                    _ => throw new ArgumentOutOfRangeException()
                },
                ///往右走
                3 => moveAction switch
                {
                    0 => this[rowCurrent, new[] {3, columnCurrent + 1}.Min()].Index,
                    1 => this[new[] {0, rowCurrent - 1}.Max(), columnCurrent].Index,
                    2 => this[new[] {3, rowCurrent + 1}.Min(), columnCurrent].Index,
                    _ => throw new ArgumentOutOfRangeException()
                },
                _ => PlayID
            };

            var state = Enumerable.Repeat(0, LakeUnits.Count).ToArray();
            state[PlayID] = 1;
            var stateTensor = torch.from_array(state, torch.ScalarType.Float32).to(Device);
            return new Observation(stateTensor);
        }

        public override float DiscountReward(Episode episode, float gamma)
        {
            var res = (float) Math.Pow(Gamma, episode.Oars.Count);
            return res;
        }

        public override bool StopEpoch(int epoch)
        {
            return this[PlayID].Role == LakeRole.End || this[PlayID].Role == LakeRole.Hole;
        }

        public override Observation Reset()
        {
            base.Reset();
            PlayID = 0;
            return Observation!;
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