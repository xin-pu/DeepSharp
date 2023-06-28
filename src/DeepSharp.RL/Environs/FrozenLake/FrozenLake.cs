using System.Text;
using DeepSharp.RL.Environs.Spaces;

namespace DeepSharp.RL.Environs
{
    public class Frozenlake : Environ<Space, Space>

    {
        public Frozenlake(int order = 4, DeviceType deviceType = DeviceType.CUDA)
            : base("Frozenlake", deviceType)
        {
            var area = (int) Math.Pow(order, 2);
            Order = order;
            LakeUnits = CreateLake4();
            PlayID = 0;
            ActionSpace = new Disperse(order, torch.ScalarType.Int32, deviceType);
            ObservationSpace = new Box(0, 1, new long[] {area}, deviceType);
            Reset();
        }

        public int Order { protected set; get; }
        public int PlayID { set; get; }

        public List<LakeUnit> LakeUnits { get; set; }

        public LakeUnit this[int i]
        {
            get => LakeUnits[i];
            set => LakeUnits[i] = value;
        }

        public LakeUnit this[int r, int c]
        {
            get => LakeUnits[r * Order + c];
            set => LakeUnits[r * Order + c] = value;
        }


        internal List<LakeUnit> CreateLake4()
        {
            var units = new List<LakeUnit>();
            foreach (var r in Enumerable.Range(0, Order))
            foreach (var c in Enumerable.Range(0, Order))
            {
                var unit = new LakeUnit(r, c, r * Order + c);
                units.Add(unit);
            }

            units[0].Role = LakeRole.Start;
            units[5].Role = units[12].Role = LakeRole.Hole;
            //units[5].Role = units[7].Role = units[11].Role = units[12].Role = LakeRole.Hole;
            units[15].Role = LakeRole.End;

            return units;
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
        ///     Player will go to target without fail.
        /// </summary>
        public void ChangeToRough()
        {
            Probs = new float[] {1, 0, 0};
        }

        /// <summary>
        ///     Means Player 1/3 => Target, 1/3 Per to Left, 1/3 Per to Right
        /// </summary>
        private float[] Probs = {1f, 1f, 1f};

        /// <summary>
        ///     When Player can't move towards the wall
        /// </summary>
        /// <returns></returns>
        public override Act SampleAct()
        {
            var rowCurrent = this[PlayID].Row;
            var columnCurrent = this[PlayID].Column;
            var probs = new List<float>
            {
                rowCurrent == 0 ? 0 : 1,
                rowCurrent == Order - 1 ? 0 : 1,
                columnCurrent == 0 ? 0 : 1,
                columnCurrent == Order - 1 ? 0 : 1
            };

            var moveProb = torch
                .multinomial(torch.from_array(probs.ToArray()), 1)
                .to_type(ActionSpace!.Type);
            return new Act(moveProb);
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
        public override Observation Update(Act act)
        {
            var banditSelectIndex = act.Value!.to_type(ActionSpace!.Type).ToInt32();

            var moveProb = torch
                .multinomial(torch.from_array(Probs), 1)
                .to_type(ActionSpace!.Type);
            var moveAction = moveProb.ToInt32();


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
            var stateTensor = torch.from_array(state, ObservationSpace!.Type).to(Device);
            return new Observation(stateTensor);
        }

        public override float DiscountReward(Episode episode, float gamma)
        {
            var res = 1f;
            return res;
        }

        public override bool IsComplete(int epoch)
        {
            return this[PlayID].Role == LakeRole.End || this[PlayID].Role == LakeRole.Hole;
        }

        public sealed override Observation Reset()
        {
            base.Reset();
            PlayID = 0;
            var state = Enumerable.Repeat(0, LakeUnits.Count).ToArray();
            state[PlayID] = 1;
            var stateTensor = torch.from_array(state, ObservationSpace!.Type).to(Device);
            var observation = Observation = new Observation(stateTensor);
            return observation;
        }

        public override string ToString()
        {
            var str = new StringBuilder();
            foreach (var r in Enumerable.Range(0, Order))
            {
                foreach (var c in Enumerable.Range(0, Order))
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