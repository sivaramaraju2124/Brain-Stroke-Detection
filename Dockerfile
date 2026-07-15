# Use the official Python image
FROM python:3.11-slim

# Set up a new non-root user (Required by Hugging Face Spaces for security)
RUN useradd -m -u 1000 user
USER user

# Set environment variables
ENV PATH="/home/user/.local/bin:$PATH"
# Hugging Face runs web apps on port 7860
ENV PORT=7860 

# Set the working directory
WORKDIR /app

# Copy requirements and install them first (for caching)
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=user:user . .

# Expose the required port
EXPOSE 7860

# Command to run the application using Gunicorn on port 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]
