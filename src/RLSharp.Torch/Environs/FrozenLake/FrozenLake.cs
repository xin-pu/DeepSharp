using System.Text;
using RLSharp.Torch.Environs.Spaces;

namespace RLSharp.Torch.Environs
{
	public class FrozenLake : EnvironmentBase<Space, Space>

	{
		public FrozenLake(int order = 4, DeviceType deviceType = DeviceType.CPU)
			: base("FrozenLake", deviceType)
		{
			var area = (int)Math.Pow(order, 2);
			Order            = order;
			LakeUnits        = CreateLake4();
			PlayID           = 0;
			ActionSpace      = new Discrete(order, torch.ScalarType.Int32, deviceType);
			ObservationSpace = new Box(0, 1, new long[] { area }, deviceType);
			Reset();
		}

		public FrozenLake(float[] soomthing, int order = 4, DeviceType deviceType = DeviceType.CPU)
			: this(order, deviceType)
		{
			Smoothing = soomthing;
		}


		/// <summary>
		///     Probability distribution for movement: 1/3 to target, 1/3 to left, 1/3 to right.
		/// </summary>
		public float[] Smoothing { get; protected set; } = { 1f, 1f, 1f };

		public int Order { get; protected set; }

		public int PlayID { get; protected set; }

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
			//units[5].Role = units[12].Role = LakeRole.Hole;
			units[5].Role  = units[7].Role = units[11].Role = units[12].Role = LakeRole.Hole;
			units[15].Role = LakeRole.End;

			return units;
		}


		/// <summary>
		///     Calculate reward from ObservationValue.
		///     Only the End position gives reward: 1.
		/// </summary>
		/// <param name="ObservationValue"></param>
		/// <returns></returns>
		public override Reward GetReward(ObservationValue ObservationValue)
		{
			var index = ObservationValue.Value!.argmax().item<long>();
			var unit  = this[(int)index];
			switch (unit.Role)
			{
				case LakeRole.End:
					return new Reward(1f);
				default:
					return new Reward(0f);
			}
		}

		/// <summary>
		///     Set movement to deterministic: player always goes to target.
		/// </summary>
		public void ChangeToRough()
		{
			Smoothing = new float[] { 1, 0, 0 };
		}

		/// <summary>
		///     Set the player's starting position.
		/// </summary>
		public void SetPlayID(int playid)
		{
			PlayID = playid;
		}


		/// <summary>
		///     Sample a valid action (prevent moving into walls).
		/// </summary>
		/// <returns></returns>
		public override ActionValue SampleAct()
		{
			var rowCurrent    = this[PlayID].Row;
			var columnCurrent = this[PlayID].Column;
			var probs = new List<float>
			{
				rowCurrent    == 0 ? 0 : 1,
				rowCurrent    == Order - 1 ? 0 : 1,
				columnCurrent == 0 ? 0 : 1,
				columnCurrent == Order - 1 ? 0 : 1
			};

			var moveProb = torch
				.multinomial(torch.from_array(probs.ToArray()), 1)
				.to_type(ActionSpace!.Type);
			return new ActionValue(moveProb);
		}


		/// <summary>
		///     Update the player position based on action.
		/// </summary>
		/// <param name="ActionValue">
		///     Action index: 0=Up, 1=Down, 2=Left, 3=Right.
		/// </param>
		/// <returns></returns>
		public override ObservationValue Update(ActionValue ActionValue)
		{
			var action = ActionValue.Value!.to_type(ActionSpace!.Type).ToInt32();

			var moveProb = torch
				.multinomial(torch.from_array(Smoothing), 1)
				.to_type(ActionSpace!.Type);
			var moveAction = moveProb.ToInt32();


			var rowCurrent    = this[PlayID].Row;
			var columnCurrent = this[PlayID].Column;

			PlayID = action switch
			{
				// Move up
				0 => moveAction switch
				{
					0 => this[new[] { 0, rowCurrent                - 1 }.Max(), columnCurrent].Index,
					1 => this[rowCurrent, new[] { 0, columnCurrent - 1 }.Max()].Index,
					2 => this[rowCurrent, new[] { 3, columnCurrent + 1 }.Min()].Index,
					_ => throw new ArgumentOutOfRangeException()
				},
				// Move down
				1 => moveAction switch
				{
					0 => this[new[] { 3, rowCurrent                + 1 }.Min(), columnCurrent].Index,
					1 => this[rowCurrent, new[] { 0, columnCurrent - 1 }.Max()].Index,
					2 => this[rowCurrent, new[] { 3, columnCurrent + 1 }.Min()].Index,
					_ => throw new ArgumentOutOfRangeException()
				},
				// Move left
				2 => moveAction switch
				{
					0 => this[rowCurrent, new[] { 0, columnCurrent - 1 }.Max()].Index,
					1 => this[new[] { 0, rowCurrent                - 1 }.Max(), columnCurrent].Index,
					2 => this[new[] { 3, rowCurrent                + 1 }.Min(), columnCurrent].Index,
					_ => throw new ArgumentOutOfRangeException()
				},
				// Move right
				3 => moveAction switch
				{
					0 => this[rowCurrent, new[] { 3, columnCurrent + 1 }.Min()].Index,
					1 => this[new[] { 0, rowCurrent                - 1 }.Max(), columnCurrent].Index,
					2 => this[new[] { 3, rowCurrent                + 1 }.Min(), columnCurrent].Index,
					_ => throw new ArgumentOutOfRangeException()
				},
				_ => PlayID
			};

			var state = Enumerable.Repeat(0, LakeUnits.Count).ToArray();
			state[PlayID] = 1;
			var stateTensor = torch.from_array(state, ObservationSpace!.Type).to(Device);
			return new ObservationValue(stateTensor);
		}


		/// <summary>
		///     Return the reward from the last step of the episode.
		/// </summary>
		/// <param name="episode"></param>
		/// <returns></returns>
		public override float GetReturn(Episode episode)
		{
			return episode.Steps.Last().Reward.Value;
		}

		public override bool IsComplete(int epoch)
		{
			return this[PlayID].Role == LakeRole.End || this[PlayID].Role == LakeRole.Hole || epoch > 100;
		}

		public sealed override ObservationValue Reset()
		{
			base.Reset();
			PlayID = 0;
			var state = Enumerable.Repeat(0, LakeUnits.Count).ToArray();
			state[PlayID] = 1;
			var stateTensor = torch.from_array(state, ObservationSpace!.Type).to(Device);
			ObservationValue = new ObservationValue(stateTensor);
			return ObservationValue;
		}

		public override string ToString()
		{
			var str = new StringBuilder();
			foreach (var r in Enumerable.Range(0, Order))
			{
				foreach (var c in Enumerable.Range(0, Order))
				{
					var lakeUnit = this[r, c];
					var P        = lakeUnit.Index == PlayID ? "P" : " ";
					var detail   = $"{lakeUnit} [{P}]";
					str.Append(detail + "\t");
				}

				str.Append("\r\n");
			}

			return str.ToString();
		}
	}
}