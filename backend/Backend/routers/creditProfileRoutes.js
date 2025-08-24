import express from "express";
import {
  createCreditProfile,
  getAllCreditProfiles,
  getCreditProfileByUser,
  updateCreditProfile,
  deleteCreditProfile,
} from "../controllers/creditProfileController.js";

const router = express.Router();

router.post("/", createCreditProfile);
router.get("/", getAllCreditProfiles);
router.get("/:userId", getCreditProfileByUser);
router.put("/:userId", updateCreditProfile);
router.delete("/:userId", deleteCreditProfile);

export default router;
