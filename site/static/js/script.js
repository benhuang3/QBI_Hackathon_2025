document.addEventListener("DOMContentLoaded", function() {
    const textarea = document.getElementById('pseq');

        // Prevent default behaviors for drag & drop
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(event => {
            textarea.addEventListener(event, function(e) {
                e.preventDefault();
                e.stopPropagation();
            });
        });

        // Highlight textarea on drag over
        textarea.addEventListener('dragover', function() {
            textarea.classList.add('dragover');
        });

        // Remove highlight when leaving
        textarea.addEventListener('dragleave', function() {
            textarea.classList.remove('dragover');
        });

        // Handle file drop
        textarea.addEventListener('drop', function(event) {
            textarea.classList.remove('dragover');
            const file = event.dataTransfer.files[0]; // Get the first file
            if (!file || file.type !== "text/plain") {
                alert("Please drop a valid .txt file.");
                return;
            }

            const reader = new FileReader();
            reader.onload = function(e) {
                textarea.value = e.target.result; // Insert file contents into textarea
            };
            reader.readAsText(file);
        });
});