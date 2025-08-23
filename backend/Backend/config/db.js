import mongoose from "mongoose";

const connectDB = async () => {
    try {
        // Attach event listener BEFORE connecting
        mongoose.connection.on('connected', () => console.log("Database Connected"));
        await mongoose.connect(`${process.env.MONGODB_URI}/nexacred`);
    } catch (error) {
        console.log(error.message);
    }
};

export default connectDB;