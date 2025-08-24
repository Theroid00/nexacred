// /controllers/creditProfileController.js
import CreditProfile from "../models/creditProfile.js";

// ✅ Create new credit profile
export const createCreditProfile = async (req, res) => {
  try {
    const creditProfile = new CreditProfile(req.body);
    await creditProfile.save();
    res.status(201).json({
      message: "Credit profile created successfully",
      creditProfile,
    });
  } catch (error) {
    res.status(400).json({ message: "Error creating credit profile", error: error.message });
  }
};

// ✅ Get all credit profiles (admin use-case)
export const getAllCreditProfiles = async (req, res) => {
  try {
    const profiles = await CreditProfile.find().populate("userId", "username email");
    res.status(200).json(profiles);
  } catch (error) {
    res.status(500).json({ message: "Error fetching profiles", error: error.message });
  }
};

// ✅ Get single credit profile by userId
export const getCreditProfileByUser = async (req, res) => {
  try {
    const { userId } = req.params;
    const profile = await CreditProfile.findOne({ userId }).populate("userId", "username email");

    if (!profile) {
      return res.status(404).json({ message: "Credit profile not found" });
    }

    res.status(200).json(profile);
  } catch (error) {
    res.status(500).json({ message: "Error fetching credit profile", error: error.message });
  }
};

// ✅ Update credit profile (auto-updates debt ratio via pre-save hook)
export const updateCreditProfile = async (req, res) => {
  try {
    const { userId } = req.params;
    const updatedProfile = await CreditProfile.findOneAndUpdate(
      { userId },
      req.body,
      { new: true, runValidators: true }
    );

    if (!updatedProfile) {
      return res.status(404).json({ message: "Credit profile not found" });
    }

    res.status(200).json({
      message: "Credit profile updated successfully",
      updatedProfile,
    });
  } catch (error) {
    res.status(400).json({ message: "Error updating credit profile", error: error.message });
  }
};

// ✅ Delete credit profile
export const deleteCreditProfile = async (req, res) => {
  try {
    const { userId } = req.params;
    const deletedProfile = await CreditProfile.findOneAndDelete({ userId });

    if (!deletedProfile) {
      return res.status(404).json({ message: "Credit profile not found" });
    }

    res.status(200).json({ message: "Credit profile deleted successfully" });
  } catch (error) {
    res.status(500).json({ message: "Error deleting credit profile", error: error.message });
  }
};
