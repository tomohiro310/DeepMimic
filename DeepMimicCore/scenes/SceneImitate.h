#pragma once

#include "scenes/RLSceneSimChar.h"
#include "anim/KinCharacter.h"

class cSceneImitate : virtual public cRLSceneSimChar
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	cSceneImitate();
	virtual ~cSceneImitate();

	virtual void ParseArgs(const std::shared_ptr<cArgParser>& parser);
	virtual void Init();

	virtual const std::shared_ptr<cKinCharacter>& GetKinChar() const;
	virtual void EnableRandRotReset(bool enable);
	virtual bool EnabledRandRotReset() const;
	virtual void EnableRandVelocityReset(bool enable);
	virtual bool EnabledRandVelocityReset() const;

	virtual double CalcReward(int agent_id) const;
	virtual eTerminate CheckTerminate(int agent_id) const;

	virtual std::string GetName() const;

protected:

	std::string mMotionFile;
	std::vector<std::string> mMotionFilesForMultiClips;
	std::shared_ptr<cKinCharacter> mKinChar;
	std::shared_ptr<cKinCharacter> mKinCharRun;
	std::shared_ptr<cKinCharacter> mKinCharWalk;

	// for multi clip
	std::vector<std::shared_ptr<cKinCharacter>> mKinCharsForMultiClips;

	Eigen::VectorXd mJointWeights;
	bool mEnableRandRotReset;
	bool mEnableRandVelocityReset;
	bool mSyncCharRootPos;
	bool mSyncCharRootRot;
	bool mEnableRootRotFail;
	double mHoldEndFrame;

	size_t mUpdateCount;

	virtual bool BuildCharacters();

	virtual void CalcJointWeights(const std::shared_ptr<cSimCharacter>& character, Eigen::VectorXd& out_weights) const;
	virtual bool BuildController(const cCtrlBuilder::tCtrlParams& ctrl_params, std::shared_ptr<cCharController>& out_ctrl);
	virtual void BuildKinChar();
	virtual bool BuildKinCharacter(int id, std::shared_ptr<cKinCharacter>& out_char) const;
	virtual void BuildKinCharsForMultiClips();
	virtual bool BuildKinCharactersForMultiClips(int id, std::shared_ptr<cKinCharacter>& out_char) const;
	virtual void UpdateCharacters(double timestep);
	virtual void UpdateKinChar(double timestep);
	virtual void SyncKinCharsForMultiClips();

	virtual void ResetCharacters();
	virtual void ResetKinChar();
	virtual void SyncCharacters();
	virtual bool EnableSyncChar() const;
	virtual void InitCharacterPosFixed(const std::shared_ptr<cSimCharacter>& out_char);

	virtual void InitJointWeights();
	virtual void ResolveCharGroundIntersect();
	virtual void ResolveCharGroundIntersect(const std::shared_ptr<cSimCharacter>& out_char) const;
	virtual void SyncKinCharRoot();
	virtual void SyncKinCharNewCycle(const cSimCharacter& sim_char, cKinCharacter& out_kin_char) const;

	virtual double GetKinTime() const;
	virtual bool CheckKinNewCycle(double timestep) const;
	virtual bool HasFallen(const cSimCharacter& sim_char) const;
	virtual bool CheckRootRotFail(const cSimCharacter& sim_char) const;
	virtual bool CheckRootRotFail(const cSimCharacter& sim_char, const cKinCharacter& kin_char) const;
	
	virtual double CalcRandKinResetTime();
	virtual double CalcRewardImitate(const cSimCharacter& sim_char, const cKinCharacter& ref_char) const;
};
