using DeepSharp.RL.Models;
using Action = DeepSharp.RL.Models.Action;

namespace DeepSharp.RL.Environs.FrozenLake
{
    public class Lake : Environ
    {
        public Lake() : base("Lake")
        {
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
            this[5].Role = this[7].Role = this[11].Role = this[12].Role = LakeRole.Hole;
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


        private Observation getObservation()
        {
            var state = Enumerable.Range(0, LakeUnits.Count).ToArray();
            state[PlayID] = 1;
            var stateTensor = torch.from_array(state);
            return new Observation(stateTensor);
        }

        /// <summary>
        ///     Change observation to Reward,
        ///     Only when at End Position will get reward:1
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public override Reward GetReward(Observation observation)
        {
            var index = observation.Value.argmax().item<int>();
            var unit = this[index];
            switch (unit.Role)
            {
                case LakeRole.End:
                    return new Reward(1);
                default:
                    return new Reward(0);
            }
        }

        public override Observation UpdateEnviron(Action action)
        {
            throw new NotImplementedException();
        }

        public override bool StopEpoch(int epoch)
        {
            return this[PlayID].Role == LakeRole.End || this[PlayID].Role == LakeRole.Hole;
        }
    }

    public class LakeUnit
    {
        /// <summary>
        /// </summary>
        /// <param name="row"></param>
        /// <param name="column"></param>
        public LakeUnit(int row, int column, int index)
        {
            Column = column;
            Row = row;
            Index = index;
            Role = LakeRole.Ice;
        }

        public int Index { set; get; }
        public int Row { set; get; }
        public int Column { set; get; }
        public LakeRole Role { set; get; }
    }

    public enum LakeRole
    {
        Ice,
        Hole,
        Start,
        End
    }
}