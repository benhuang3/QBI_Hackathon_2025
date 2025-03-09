import React, { useState } from "react";

const App = () => {
    const [num1, setNum1] = useState("");
    const [num2, setNum2] = useState("");
    const [result, setResult] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();

        const response = await fetch("http://127.0.0.1:5000/calculate", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ num1, num2 }),
        });

        const data = await response.json();
        setResult(data.sum);
    };

    return (
        <div style={{ textAlign: "center", marginTop: "50px" }}>
            <h1>React & Flask Sum Calculator</h1>
            <form onSubmit={handleSubmit}>
                <input
                    type="number"
                    placeholder="Enter first number"
                    value={num1}
                    onChange={(e) => setNum1(e.target.value)}
                    style={{ padding: "10px", margin: "10px", fontSize: "18px" }}
                />
                <input
                    type="number"
                    placeholder="Enter second number"
                    value={num2}
                    onChange={(e) => setNum2(e.target.value)}
                    style={{ padding: "10px", margin: "10px", fontSize: "18px" }}
                />
                <button
                    type="submit"
                    style={{
                        padding: "15px 30px",
                        fontSize: "20px",
                        backgroundColor: "#007bff",
                        color: "white",
                        border: "none",
                        cursor: "pointer",
                        borderRadius: "5px",
                        marginTop: "20px",
                    }}
                >
                    Calculate
                </button>
            </form>

            {result !== null && <h2 style={{ marginTop: "20px" }}>Sum: {result}</h2>}
        </div>
    );
};

export default App;
